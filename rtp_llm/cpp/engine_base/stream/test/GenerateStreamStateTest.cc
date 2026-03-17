
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class GenerateStreamStateTest: public DeviceTestBase {
protected:
    GenerateStreamStateTest(): perf_scope("PERF_TEST", "1") {}

    CacheConfig init_config() {
        return test::makeSimpleMhaCacheConfig(
            /*layer_num=*/3, /*block_num=*/9, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
    }

    GenerateStreamPtr createStream(const std::vector<int>& input_tokens = {1, 2, 3, 4, 5, 6},
                                   bool                    reuse_cache  = false) {
        cache_manager_ =
            std::make_shared<KVCacheManager>(init_config(), device_, /*warmup=*/false, /*metrics_reporter=*/nullptr);
        EXPECT_TRUE(cache_manager_->init());
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager_;
        resource_context.reuse_cache   = reuse_cache;

        std::shared_ptr<GenerateInput>  generate_input(new GenerateInput());
        std::shared_ptr<GenerateConfig> generate_config(new GenerateConfig());
        generate_config->num_return_sequences = 1;
        auto                vec               = input_tokens;
        std::vector<size_t> shape             = {vec.size()};
        generate_input->input_ids =
            std::make_unique<rtp_llm::Buffer>(rtp_llm::MEMORY_CPU, rtp_llm::TYPE_INT32, shape, (void*)(vec.data()));
        generate_input->generate_config = generate_config;
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = 2048;
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);
    }

protected:
    autil::EnvGuard                 perf_scope;
    std::shared_ptr<KVCacheManager> cache_manager_;
};

// ============================================================================
// 1. LOADING_CACHE state and state transitions
// ============================================================================

TEST_F(GenerateStreamStateTest, testInitialStateIsWaiting) {
    auto stream = createStream();
    ASSERT_TRUE(stream->waiting());
    ASSERT_FALSE(stream->finished());
    ASSERT_FALSE(stream->running());
    ASSERT_FALSE(stream->loadingCache());
}

TEST_F(GenerateStreamStateTest, testSetLoadingCacheFromWaiting) {
    auto stream = createStream();
    // WAITING -> LOADING_CACHE should succeed
    ASSERT_TRUE(stream->setLoadingCache());
    ASSERT_TRUE(stream->loadingCache());
    ASSERT_FALSE(stream->waiting());
    ASSERT_FALSE(stream->running());
    ASSERT_FALSE(stream->finished());
    ASSERT_EQ(stream->generate_status_->status, StreamState::LOADING_CACHE);
}

TEST_F(GenerateStreamStateTest, testSetLoadingCacheFromNonWaitingFails) {
    auto stream = createStream();
    // Set to RUNNING first
    ASSERT_TRUE(stream->setRunning());
    // RUNNING -> LOADING_CACHE should fail
    ASSERT_FALSE(stream->setLoadingCache());
    ASSERT_TRUE(stream->running());

    // FINISHED -> LOADING_CACHE should fail
    auto stream2 = createStream();
    stream2->setFinishedWithoutLock();
    ASSERT_FALSE(stream2->setLoadingCache());
    ASSERT_TRUE(stream2->finished());
}

TEST_F(GenerateStreamStateTest, testSetWaitingFromLoadingCache) {
    auto stream = createStream();
    ASSERT_TRUE(stream->setLoadingCache());
    // LOADING_CACHE -> WAITING should succeed
    ASSERT_TRUE(stream->setWaiting());
    ASSERT_TRUE(stream->waiting());
    ASSERT_FALSE(stream->loadingCache());
}

TEST_F(GenerateStreamStateTest, testSetWaitingFromNonLoadingCacheFails) {
    auto stream = createStream();
    // WAITING -> WAITING should fail (setWaiting only works from LOADING_CACHE)
    ASSERT_FALSE(stream->setWaiting());
    ASSERT_TRUE(stream->waiting());

    // RUNNING -> WAITING should fail
    auto stream2 = createStream();
    stream2->setRunning();
    ASSERT_FALSE(stream2->setWaiting());
}

// ============================================================================
// 2. WAITING -> LOADING_CACHE -> WAITING/FINISHED paths
// ============================================================================

TEST_F(GenerateStreamStateTest, testWaitingToLoadingCacheToWaitingPath) {
    auto stream = createStream();
    // WAITING -> LOADING_CACHE
    ASSERT_TRUE(stream->setLoadingCache());
    ASSERT_TRUE(stream->loadingCache());
    // LOADING_CACHE -> WAITING (load done, back to waiting for scheduling)
    ASSERT_TRUE(stream->setWaiting());
    ASSERT_TRUE(stream->waiting());
    // WAITING -> RUNNING (normal scheduling)
    ASSERT_TRUE(stream->setRunning());
    ASSERT_TRUE(stream->running());
}

TEST_F(GenerateStreamStateTest, testWaitingToLoadingCacheToFinishedViaCancel) {
    auto stream = createStream();
    // WAITING -> LOADING_CACHE
    ASSERT_TRUE(stream->setLoadingCache());
    ASSERT_TRUE(stream->loadingCache());
    // Cancel while in LOADING_CACHE
    stream->cancelIfNotRunning();
    ASSERT_TRUE(stream->finished());
    ASSERT_EQ(stream->generate_status_->error_info.code(), ErrorCode::CANCELLED);
}

TEST_F(GenerateStreamStateTest, testWaitingToLoadingCacheToFinishedViaSetStop) {
    auto stream = createStream();
    ASSERT_TRUE(stream->setLoadingCache());
    // setStop during LOADING_CACHE transitions to FINISHED
    stream->setStop(ErrorCode::GENERATE_TIMEOUT, "timeout during loading");
    ASSERT_TRUE(stream->finished());
    ASSERT_TRUE(stream->hasError());
    ASSERT_EQ(stream->generate_status_->error_info.code(), ErrorCode::GENERATE_TIMEOUT);
}

// ============================================================================
// 3. RUNNING -> FINISHED / REMOTE_RUNNING transitions
// ============================================================================

TEST_F(GenerateStreamStateTest, testRunningToFinishedNormal) {
    auto stream = createStream();
    ASSERT_TRUE(stream->setRunning());
    stream->setFinishedWithoutLock();
    ASSERT_TRUE(stream->finished());
    ASSERT_FALSE(stream->running());
}

TEST_F(GenerateStreamStateTest, testRunningToFinishedViaSetStop) {
    auto stream = createStream();
    ASSERT_TRUE(stream->setRunning());
    stream->setStop(ErrorCode::MALLOC_FAILED, "OOM");
    ASSERT_TRUE(stream->finished());
    ASSERT_TRUE(stream->hasError());
}

TEST_F(GenerateStreamStateTest, testRunningToRemoteRunning) {
    auto stream = createStream();
    ASSERT_TRUE(stream->setRunning());
    stream->setNeedRemoteGenerate(true);
    ASSERT_TRUE(stream->setRemoteGenerate());
    ASSERT_TRUE(stream->isRemoteRunningWithoutLock());
    ASSERT_FALSE(stream->running());
}

TEST_F(GenerateStreamStateTest, testSetRemoteGenerateFailsIfFinished) {
    auto stream = createStream();
    stream->setFinishedWithoutLock();
    ASSERT_FALSE(stream->setRemoteGenerate());
}

TEST_F(GenerateStreamStateTest, testSetRunningFailsIfFinished) {
    auto stream = createStream();
    stream->setFinishedWithoutLock();
    ASSERT_FALSE(stream->setRunning());
}

// ============================================================================
// 4. reportError / hasError / statusInfo
// ============================================================================

TEST_F(GenerateStreamStateTest, testReportErrorDoesNotChangeState) {
    auto stream = createStream();
    ASSERT_TRUE(stream->waiting());
    // reportError only stores error, does NOT change state
    stream->reportError(ErrorCode::CANCELLED, "cancel from RPC");
    ASSERT_TRUE(stream->hasError());
    ASSERT_TRUE(stream->waiting());  // state unchanged
    ASSERT_EQ(stream->generate_status_->status, StreamState::WAITING);
}

TEST_F(GenerateStreamStateTest, testReportErrorFirstWins) {
    auto stream = createStream();
    stream->reportError(ErrorCode::CANCELLED, "first error");
    stream->reportError(ErrorCode::GENERATE_TIMEOUT, "second error");
    auto err = stream->statusInfo();
    // First error should win
    ASSERT_EQ(err.code(), ErrorCode::CANCELLED);
}

TEST_F(GenerateStreamStateTest, testSetStopChangesStateToFinished) {
    auto stream = createStream();
    ASSERT_TRUE(stream->waiting());
    stream->setStop(ErrorCode::MALLOC_FAILED, "OOM");
    ASSERT_TRUE(stream->finished());
    ASSERT_TRUE(stream->hasError());
    ASSERT_EQ(stream->generate_status_->status, StreamState::FINISHED);
}

TEST_F(GenerateStreamStateTest, testStopAndReleaseChangesStateAndReleasesResource) {
    auto stream = createStream();
    stream->setRunning();
    auto& resource = stream->streamCacheResource();
    ASSERT_TRUE(resource.initKVBlock().ok());
    ASSERT_GT(resource.curBlocksNum(), 0);

    stream->stopAndRelease(ErrorCode::MALLOC_FAILED, "OOM");
    ASSERT_TRUE(stream->finished());
    ASSERT_TRUE(stream->hasError());
}

// ============================================================================
// 5. cancelIfNotRunning behavior for all non-running states
// ============================================================================

TEST_F(GenerateStreamStateTest, testCancelIfNotRunning_Waiting) {
    auto stream = createStream();
    ASSERT_TRUE(stream->waiting());
    stream->cancelIfNotRunning();
    ASSERT_TRUE(stream->finished());
    ASSERT_EQ(stream->generate_status_->error_info.code(), ErrorCode::CANCELLED);
}

TEST_F(GenerateStreamStateTest, testCancelIfNotRunning_LoadingCache) {
    auto stream = createStream();
    ASSERT_TRUE(stream->setLoadingCache());
    stream->cancelIfNotRunning();
    ASSERT_TRUE(stream->finished());
    ASSERT_EQ(stream->generate_status_->error_info.code(), ErrorCode::CANCELLED);
}

TEST_F(GenerateStreamStateTest, testCancelIfNotRunning_Running_NoEffect) {
    auto stream = createStream();
    ASSERT_TRUE(stream->setRunning());
    stream->cancelIfNotRunning();
    // Should NOT cancel a running stream
    ASSERT_TRUE(stream->running());
    ASSERT_FALSE(stream->finished());
}

TEST_F(GenerateStreamStateTest, testCancelIfNotRunning_RemoteRunning) {
    auto stream = createStream();
    ASSERT_TRUE(stream->setRunning());
    ASSERT_TRUE(stream->setRemoteGenerate());
    stream->cancelIfNotRunning();
    ASSERT_TRUE(stream->finished());
    ASSERT_EQ(stream->generate_status_->error_info.code(), ErrorCode::CANCELLED);
}

// ============================================================================
// 6. cancel() uses reportError (external error reporting)
// ============================================================================

TEST_F(GenerateStreamStateTest, testCancelUsesReportError) {
    auto stream = createStream();
    ASSERT_TRUE(stream->setRunning());
    stream->cancel();
    // cancel() calls reportError(), not setStop(), so state is unchanged
    ASSERT_TRUE(stream->hasError());
    ASSERT_TRUE(stream->running());  // State NOT changed by cancel()
    auto err = stream->statusInfo();
    ASSERT_EQ(err.code(), ErrorCode::CANCELLED);
}

// ============================================================================
// 7. Complete state machine cycle: WAITING -> LOADING_CACHE -> WAITING -> RUNNING -> FINISHED
// ============================================================================

TEST_F(GenerateStreamStateTest, testFullStateMachineCycle) {
    auto stream = createStream();
    // Initial: WAITING
    ASSERT_EQ(stream->generate_status_->status, StreamState::WAITING);

    // WAITING -> LOADING_CACHE
    ASSERT_TRUE(stream->setLoadingCache());
    ASSERT_EQ(stream->generate_status_->status, StreamState::LOADING_CACHE);

    // LOADING_CACHE -> WAITING (load done)
    ASSERT_TRUE(stream->setWaiting());
    ASSERT_EQ(stream->generate_status_->status, StreamState::WAITING);

    // WAITING -> RUNNING
    ASSERT_TRUE(stream->setRunning());
    ASSERT_EQ(stream->generate_status_->status, StreamState::RUNNING);

    // RUNNING -> FINISHED
    stream->setFinishedWithoutLock();
    ASSERT_EQ(stream->generate_status_->status, StreamState::FINISHED);
}

TEST_F(GenerateStreamStateTest, testFullStateMachineCycleWithRemoteRunning) {
    auto stream = createStream();
    // WAITING -> LOADING_CACHE -> WAITING -> RUNNING -> REMOTE_RUNNING -> FINISHED
    ASSERT_TRUE(stream->setLoadingCache());
    ASSERT_TRUE(stream->setWaiting());
    ASSERT_TRUE(stream->setRunning());
    ASSERT_TRUE(stream->setRemoteGenerate());
    ASSERT_EQ(stream->generate_status_->status, StreamState::REMOTE_RUNNING);
    stream->setFinishedWithoutLock();
    ASSERT_TRUE(stream->finished());
}

// ============================================================================
// 8. isDoneWithoutLock checks
// ============================================================================

TEST_F(GenerateStreamStateTest, testIsDoneWithoutLock) {
    auto stream = createStream();
    // sub_generate_status_ is initialized, check batch_id=0
    ASSERT_FALSE(stream->isDoneWithoutLock(0));

    stream->setFinishedWithoutLock();
    ASSERT_TRUE(stream->isDoneWithoutLock(0));
}

// ============================================================================
// 9. StreamStateToString
// ============================================================================

TEST_F(GenerateStreamStateTest, testStreamStateToString) {
    ASSERT_EQ(StreamStateToString(StreamState::WAITING), "WAITING");
    ASSERT_EQ(StreamStateToString(StreamState::LOADING_CACHE), "LOADING_CACHE");
    ASSERT_EQ(StreamStateToString(StreamState::RUNNING), "RUNNING");
    ASSERT_EQ(StreamStateToString(StreamState::FINISHED), "FINISHED");
    ASSERT_EQ(StreamStateToString(StreamState::REMOTE_RUNNING), "REMOTE_RUNNING");
}

}  // namespace rtp_llm
