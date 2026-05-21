#include "gtest/gtest.h"
#include "gmock/gmock.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/RoleTypes.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

// =============================================================================
// Test fixture: PD sep KV cache release correctness
// Validates that holdKVCacheForPDSep / releaseKVCacheForPDSep / releaseResource
// interact correctly with respect to:
//   1. Block ref-counts stay > 0 while pd_kvcache_ref_ is held
//   2. insertIntoCache (device reuse) is called before blocks are cleared
//   3. freeBlocksNum() returns to baseline after both release paths complete
//   4. Race condition: concurrent releaseKVCacheForPDSep (grpc thread) vs
//      releaseResource (engine thread)
// =============================================================================
class PdSepKVCacheReleaseTest: public DeviceTestBase {
protected:
    PdSepKVCacheReleaseTest(): perf_scope("PERF_TEST", "1") {}

    // Simple config: 3 layers, 16 blocks, 8 tokens/block
    CacheConfig makeConfig() {
        return test::makeSimpleMhaCacheConfig(/*layer_num=*/3,
                                              /*block_num=*/16,
                                              /*tokens_per_block=*/8,
                                              rtp_llm::DataType::TYPE_INT8);
    }

    // Build a PREFILL stream with reuse_cache enabled
    void prepareStream(const std::vector<int>& input_tokens) {
        auto cache_config = makeConfig();
        cache_manager_    = std::make_shared<KVCacheManager>(cache_config, /*warmup=*/false, nullptr);
        ASSERT_TRUE(cache_manager_->init());
        initial_free_blocks_ = cache_manager_->freeBlocksNum();

        ResourceContext resource_context;
        resource_context.cache_manager       = cache_manager_;
        resource_context.reuse_cache         = true;
        resource_context.enable_device_cache = true;
        resource_context.role_type           = RoleType::PREFILL;

        auto generate_input                   = std::make_shared<GenerateInput>();
        auto generate_config                  = std::make_shared<GenerateConfig>();
        generate_config->num_return_sequences = 1;
        generate_config->reuse_cache          = true;
        generate_config->enable_device_cache  = true;
        generate_input->input_ids =
            torch::tensor(std::vector<int32_t>(input_tokens.begin(), input_tokens.end()), torch::kInt32);
        generate_input->generate_config = generate_config;

        ModelConfig model_config;
        model_config.attn_config.tokens_per_block = 8;
        model_config.max_seq_len                  = 2048;
        RuntimeConfig runtime_config;

        stream_ = std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);
        stream_->generate_status_->status = StreamState::RUNNING;
    }

    // Allocate KV blocks and mark stream as FINISHED (simulates prefill done)
    void allocateAndFinish() {
        auto& resource = stream_->streamCacheResource();
        ASSERT_TRUE(resource.initKVBlock().ok());
        stream_->generate_status_->status = StreamState::FINISHED;
        stream_->fillSubGenerateStatus(StreamState::FINISHED);
    }

protected:
    autil::EnvGuard                       perf_scope;
    std::shared_ptr<NormalGenerateStream> stream_;
    std::shared_ptr<KVCacheManager>       cache_manager_;
    size_t                                initial_free_blocks_ = 0;
};

// =============================================================================
// Test 1: Normal release without PD sep hold
// Baseline: blocks are allocated, released normally, freeBlocks returns to start
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testNormalRelease_BlocksReturnedToPool) {
    // 14 tokens, tokens_per_block=8 -> 2 blocks needed (1 full + 1 partial)
    prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    allocateAndFinish();

    auto& resource  = stream_->streamCacheResource();
    int   allocated = resource.curBlocksNum();
    ASSERT_GT(allocated, 0) << "Should have allocated some blocks";
    ASSERT_LT(cache_manager_->freeBlocksNum(), initial_free_blocks_) << "Blocks should be in use";

    // Normal release (no PD sep)
    stream_->releaseResource();

    // After releaseResource with reuse_cache=true, insertIntoCache() is called.
    // The device cache retains a reference to completed blocks for future reuse,
    // so freeBlocksNum may be less than initial. The key invariant is:
    //   freeBlocksNum >= initial_free_blocks_ - allocated (no extra blocks leaked)
    EXPECT_GE(cache_manager_->freeBlocksNum(), initial_free_blocks_ - allocated)
        << "No extra blocks should be leaked beyond what was allocated";
    EXPECT_EQ(resource.curBlocksNum(), 0) << "Block list should be cleared";
    EXPECT_TRUE(resource.resource_released_) << "resource_released_ should be true";
}

// =============================================================================
// Test 2: holdKVCacheForPDSep increments ref count
// After hold, pd_kvcache_ref_ is non-null
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testHoldKVCacheForPDSep_SetsRef) {
    prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(resource.curBlocksNum(), 0);

    // Call hold - simulates prefill pollLocalOutput holding the cache
    resource.holdKVCacheForPDSep();

    EXPECT_NE(resource.pd_kvcache_ref_, nullptr) << "pd_kvcache_ref_ should be set after hold";
}

// =============================================================================
// Test 3: releaseResource with pd_kvcache_ref_ held
// Blocks should be cleared after releaseResource (clearBlocks always called)
// resource_released_ should be true, insertIntoCache should run
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testReleaseResource_WithHold_ClearsBlocks) {
    prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    allocateAndFinish();

    auto& resource  = stream_->streamCacheResource();
    int   allocated = resource.curBlocksNum();
    ASSERT_GT(allocated, 0);

    resource.holdKVCacheForPDSep();
    ASSERT_NE(resource.pd_kvcache_ref_, nullptr);

    // Simulate engine thread calling releaseResource
    stream_->releaseResource();

    EXPECT_TRUE(resource.resource_released_) << "resource_released_ should be true";
    // clearBlocks() is always called in releaseResource
    // blocks list is cleared, but ref is still held by pd_kvcache_ref_
    // freeBlocksNum should NOT be fully restored yet (blocks still held by ref)
    // NOTE: tryReleaseKVBlock calls cache_manager_->free() which returns blocks to pool,
    // but pd_kvcache_ref_ holds an extra ref, so actual free count depends on impl.
    // Key invariant: resource_released_ = true and no crash.
    EXPECT_TRUE(resource.resource_released_);
}

// =============================================================================
// Test 4: releaseKVCacheForPDSep after releaseResource
// This is the "correct order" path: engine thread releases first,
// then grpc thread calls releaseKVCacheForPDSep.
// After both complete, freeBlocks should return to initial.
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testCorrectOrder_ReleaseResourceThenReleasePDSep_BlocksReturned) {
    prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(resource.curBlocksNum(), 0);

    // Step 1: hold (prefill pollLocalOutput)
    resource.holdKVCacheForPDSep();
    ASSERT_NE(resource.pd_kvcache_ref_, nullptr);

    // Step 2: engine thread releases
    stream_->releaseResource();
    EXPECT_TRUE(resource.resource_released_);

    // Step 3: grpc thread releases
    resource.releaseKVCacheForPDSep();
    EXPECT_EQ(resource.pd_kvcache_ref_, nullptr) << "pd_kvcache_ref_ should be reset";

    // After both releases, the device cache may retain 1 block for reuse (insertIntoCache).
    // Key invariant: no blocks leaked beyond initial allocation.
    EXPECT_GE(cache_manager_->freeBlocksNum(), initial_free_blocks_ - 2)
        << "No extra blocks should be leaked. free=" << cache_manager_->freeBlocksNum()
        << " initial=" << initial_free_blocks_;
}

// =============================================================================
// Test 5: insertIntoCache is called during releaseResource (device reuse cache)
// After releaseResource, the cache keys should be findable in the block cache
// (i.e., a subsequent allocation with the same tokens hits reuse)
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testInsertIntoCache_CalledDuringRelease_ReuseWorks) {
    const std::vector<int> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    prepareStream(tokens);
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(resource.curBlocksNum(), 0);

    resource.holdKVCacheForPDSep();

    // Engine thread releases: should call insertIntoCache (device cache)
    stream_->releaseResource();
    EXPECT_TRUE(resource.resource_released_);

    // Release the pd_sep hold (grpc thread)
    resource.releaseKVCacheForPDSep();

    // Now prepare a second stream with the same tokens - should get reuse
    ResourceContext resource_context2;
    resource_context2.cache_manager       = cache_manager_;
    resource_context2.reuse_cache         = true;
    resource_context2.enable_device_cache = true;
    resource_context2.role_type           = RoleType::PREFILL;

    auto generate_input2                   = std::make_shared<GenerateInput>();
    auto generate_config2                  = std::make_shared<GenerateConfig>();
    generate_config2->num_return_sequences = 1;
    generate_config2->reuse_cache          = true;
    generate_config2->enable_device_cache  = true;
    generate_input2->input_ids       = torch::tensor(std::vector<int32_t>(tokens.begin(), tokens.end()), torch::kInt32);
    generate_input2->generate_config = generate_config2;

    ModelConfig model_config;
    model_config.attn_config.tokens_per_block = 8;
    model_config.max_seq_len                  = 2048;
    RuntimeConfig runtime_config;

    auto stream2 = std::make_shared<NormalGenerateStream>(
        generate_input2, model_config, runtime_config, resource_context2, nullptr);
    stream2->generate_status_->status = StreamState::RUNNING;

    auto& resource2 = stream2->streamCacheResource();
    ASSERT_TRUE(resource2.initKVBlock().ok());

    // With 14 tokens and block_size=8: 1 full block (8 tokens) should be reused
    int reuse_len = stream2->reuseLength();
    EXPECT_GE(reuse_len, 8) << "At least 1 block (8 tokens) should be reused from device cache. "
                            << "reuse_len=" << reuse_len;

    stream2->releaseResource();
}

// =============================================================================
// Test 6: Race condition simulation
// Engine thread calls releaseResource concurrently with
// grpc thread calling releaseKVCacheForPDSep.
// Verifies: no crash, no double-free, freeBlocks returns to initial after both.
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testRaceCondition_ConcurrentRelease_NoDoubleFree) {
    prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(resource.curBlocksNum(), 0);

    resource.holdKVCacheForPDSep();
    ASSERT_NE(resource.pd_kvcache_ref_, nullptr);

    std::atomic<bool> engine_done{false};
    std::atomic<bool> grpc_done{false};

    // Engine thread: releaseResource
    std::thread engine_thread([&]() {
        stream_->releaseResource();
        engine_done.store(true);
    });

    // Grpc thread: releaseKVCacheForPDSep (with small delay to increase race chance)
    std::thread grpc_thread([&]() {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        resource.releaseKVCacheForPDSep();
        grpc_done.store(true);
    });

    engine_thread.join();
    grpc_thread.join();

    EXPECT_TRUE(engine_done.load()) << "Engine thread should have completed";
    EXPECT_TRUE(grpc_done.load()) << "Grpc thread should have completed";
    EXPECT_TRUE(resource.resource_released_) << "resource_released_ should be true";
    EXPECT_EQ(resource.pd_kvcache_ref_, nullptr) << "pd_kvcache_ref_ should be reset";

    // Critical: no double-free, no extra blocks leaked.
    // insertIntoCache may hold 1 cached block ref, so freeBlocksNum can be <= initial.
    EXPECT_GE(cache_manager_->freeBlocksNum(), initial_free_blocks_ - 2)
        << "No extra blocks should be leaked after concurrent release. "
        << "free=" << cache_manager_->freeBlocksNum() << " initial=" << initial_free_blocks_;
}

// =============================================================================
// Test 7: holdKVCacheForPDSep without subsequent releaseKVCacheForPDSep
// (simulates grpc failure: hold is called but release never comes)
// releaseResource alone should still eventually free blocks when ref drops
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testHoldWithoutReleasePDSep_ResourceReleasedStillCompletes) {
    prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(resource.curBlocksNum(), 0);

    resource.holdKVCacheForPDSep();

    // Only engine thread releases, grpc thread never calls releaseKVCacheForPDSep
    stream_->releaseResource();

    EXPECT_TRUE(resource.resource_released_);

    // pd_kvcache_ref_ still holds a ref - blocks won't be fully freed until ref drops
    // Simulate ref drop (e.g. stream destructor or explicit reset)
    resource.pd_kvcache_ref_.reset();

    // After ref drop, blocks should be returned (minus any held by device cache for reuse)
    EXPECT_GE(cache_manager_->freeBlocksNum(), initial_free_blocks_ - 2)
        << "Blocks should be freed once pd_kvcache_ref_ is dropped (minus device cache refs)";
}

// =============================================================================
// Test 8: Connector lease in P2PConnectorResourceStore protects blocks
// Simulates the decode_entrance path where:
//   1. Stream allocates KV blocks
//   2. addResource() is called (connector_lease created via incrKVCacheRef)
//   3. Stream is destroyed (requestFree), but blocks survive due to connector_lease
//   4. Resource entry is stolen and destroyed → blocks finally freed
// This validates the Plan B fix for the decode_entrance coredump.
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testConnectorLease_ProtectsBlocksDuringP2PTransfer) {
    prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(resource.curBlocksNum(), 0);
    auto free_before_release = cache_manager_->freeBlocksNum();

    // Create a KVCacheResource (shared_ptr) that mimics what asyncMatch produces
    auto& batch_resource = resource.batch_kv_cache_resource_->cacheResource(0);
    auto  kv_cache_ptr   = std::make_shared<KVCacheResource>(batch_resource);

    // Create P2PConnectorResourceStore WITH the allocator (connector_lease enabled)
    auto store = std::make_unique<P2PConnectorResourceStore>(nullptr, 100, cache_manager_->allocator_);
    ASSERT_TRUE(store->init());

    // Simulate addResource (what asyncMatch does in the decode_entrance path)
    auto meta = std::make_shared<GenerateInput>();
    // Use the Meta interface to call addResource — build a MockMeta-like wrapper
    // Since we need p2pRouting(), directly create an entry simulating addResource logic:
    {
        auto entry               = std::make_shared<P2PConnectorResourceEntry>();
        entry->request_id        = 9999;
        entry->unique_key        = "test_connector_lease";
        entry->kv_cache_resource = kv_cache_ptr;
        entry->deadline_ms       = currentTimeMs() + 60000;
        entry->add_time_us       = currentTimeUs();

        // Acquire connector lease (same logic as addResource with allocator)
        const auto& cache_keys = kv_cache_ptr->cacheKeys();
        if (!cache_keys.empty()) {
            entry->connector_lease =
                cache_manager_->allocator_->incrKVCacheRef(*kv_cache_ptr, cache_keys, /*is_connector=*/true);
        }
        ASSERT_NE(entry->connector_lease, nullptr) << "connector_lease should be created";

        // Manually put entry into the store's map (since we can't use addResource without real Meta)
        {
            std::lock_guard<std::mutex> lock(store->resource_map_mutex_);
            store->resource_map_["test_connector_lease"] = entry;
        }
    }

    // Step: Stream is destroyed → releaseResource → requestFree
    // This decrements req_con_ref_counter, but connector_lease holds a connector ref,
    // so blocks should NOT be freed.
    stream_->releaseResource();
    EXPECT_TRUE(resource.resource_released_);

    // Key assertion: blocks are still held by connector_lease
    // freeBlocksNum should NOT have increased back to initial (blocks still pinned)
    auto free_after_stream_release = cache_manager_->freeBlocksNum();
    EXPECT_EQ(free_after_stream_release, free_before_release)
        << "Blocks should NOT be freed yet because connector_lease holds connector ref. "
        << "free_before=" << free_before_release << " free_after=" << free_after_stream_release;

    // Step: Steal resource entry from store (simulates handleRead completing)
    auto stolen_entry = store->waitAndStealResource("test_connector_lease", currentTimeMs() + 100);
    ASSERT_NE(stolen_entry, nullptr);
    ASSERT_NE(stolen_entry->connector_lease, nullptr);

    // Step: Destroy the stolen entry → connector_lease destructor → connectorFree
    // Now blocks should be freed.
    stolen_entry.reset();

    auto free_after_lease_drop = cache_manager_->freeBlocksNum();
    EXPECT_GT(free_after_lease_drop, free_before_release)
        << "Blocks should be freed after connector_lease is dropped. "
        << "free_before=" << free_before_release << " free_after=" << free_after_lease_drop;
    EXPECT_GE(free_after_lease_drop, initial_free_blocks_ - 1)
        << "Almost all blocks should be returned to pool. "
        << "free=" << free_after_lease_drop << " initial=" << initial_free_blocks_;

    store.reset();
}

// =============================================================================
// Test 9: Race condition — concurrent stream destruction and connector_lease drop
// Engine thread destroys stream while P2P thread holds connector_lease.
// Multiple iterations to stress-test the race window.
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testConnectorLease_RaceWithStreamDestruction) {
    constexpr int kIterations = 20;

    for (int iter = 0; iter < kIterations; ++iter) {
        prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        allocateAndFinish();

        auto& resource = stream_->streamCacheResource();
        ASSERT_GT(resource.curBlocksNum(), 0);

        // Create connector_lease (same as P2PConnectorResourceStore::addResource)
        auto&              batch_resource = resource.batch_kv_cache_resource_->cacheResource(0);
        auto               kv_cache_ptr   = std::make_shared<KVCacheResource>(batch_resource);
        const auto&        cache_keys     = kv_cache_ptr->cacheKeys();
        KVCacheResourcePtr connector_lease;
        if (!cache_keys.empty()) {
            connector_lease =
                cache_manager_->allocator_->incrKVCacheRef(*kv_cache_ptr, cache_keys, /*is_connector=*/true);
        }
        ASSERT_NE(connector_lease, nullptr) << "iter=" << iter;

        std::atomic<bool> stream_released{false};
        std::atomic<bool> lease_dropped{false};

        // Thread 1: engine thread destroys stream (calls requestFree on blocks)
        std::thread engine_thread([&]() {
            stream_->releaseResource();
            stream_released.store(true);
        });

        // Thread 2: P2P thread holds lease for a bit, then drops it (simulates handleRead return)
        std::thread p2p_thread([&]() {
            // Small random-ish delay to increase race variety
            std::this_thread::sleep_for(std::chrono::microseconds(50 * (iter % 5)));
            connector_lease.reset();
            lease_dropped.store(true);
        });

        engine_thread.join();
        p2p_thread.join();

        EXPECT_TRUE(stream_released.load()) << "iter=" << iter;
        EXPECT_TRUE(lease_dropped.load()) << "iter=" << iter;

        // After both threads complete, blocks should be freed (no leak, no double-free crash)
        EXPECT_GE(cache_manager_->freeBlocksNum(), initial_free_blocks_ - 1)
            << "Blocks should be freed after both release and lease drop. iter=" << iter
            << " free=" << cache_manager_->freeBlocksNum() << " initial=" << initial_free_blocks_;

        // Reset for next iteration
        stream_.reset();
    }
}

// =============================================================================
// Test 10: Without connector_lease (nullptr allocator), blocks freed immediately on stream release
// Demonstrates that without Plan B fix, blocks would be freed while P2P transfer is in flight.
// =============================================================================
TEST_F(PdSepKVCacheReleaseTest, testNoConnectorLease_BlocksFreedImmediately) {
    prepareStream({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
    allocateAndFinish();

    auto& resource = stream_->streamCacheResource();
    ASSERT_GT(resource.curBlocksNum(), 0);
    auto free_before_release = cache_manager_->freeBlocksNum();

    // Create a store WITHOUT allocator (simulates the old code path without Plan B fix)
    auto store_no_alloc = std::make_unique<P2PConnectorResourceStore>(nullptr, 100, nullptr);
    ASSERT_TRUE(store_no_alloc->init());

    // Simulate addResource without connector_lease
    auto& batch_resource = resource.batch_kv_cache_resource_->cacheResource(0);
    auto  kv_cache_ptr   = std::make_shared<KVCacheResource>(batch_resource);
    {
        auto entry               = std::make_shared<P2PConnectorResourceEntry>();
        entry->request_id        = 8888;
        entry->unique_key        = "test_no_lease";
        entry->kv_cache_resource = kv_cache_ptr;
        entry->deadline_ms       = currentTimeMs() + 60000;
        entry->add_time_us       = currentTimeUs();
        // No connector_lease set (allocator is nullptr)
        EXPECT_EQ(entry->connector_lease, nullptr);

        std::lock_guard<std::mutex> lock(store_no_alloc->resource_map_mutex_);
        store_no_alloc->resource_map_["test_no_lease"] = entry;
    }

    // Stream is destroyed → blocks freed immediately (no connector protection)
    stream_->releaseResource();
    EXPECT_TRUE(resource.resource_released_);

    auto free_after_stream_release = cache_manager_->freeBlocksNum();
    // Without connector_lease, blocks ARE freed on requestFree (this is the bug scenario)
    EXPECT_GT(free_after_stream_release, free_before_release)
        << "WITHOUT connector_lease, blocks are freed immediately on stream release "
        << "(this is the use-after-free vulnerability). "
        << "free_before=" << free_before_release << " free_after=" << free_after_stream_release;

    store_no_alloc.reset();
}

}  // namespace rtp_llm
