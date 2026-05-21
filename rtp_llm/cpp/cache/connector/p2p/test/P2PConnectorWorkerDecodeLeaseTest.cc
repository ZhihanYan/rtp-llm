#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/connector/p2p/DecodeTargetWriteLease.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerDecode.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace test {

// ---------------------------------------------------------------------------
// InflightMockRecvTask — simulates real TransferTask cancel() semantics:
//   PENDING  → cancel() → done immediately
//   TRANSFERRING → cancel() → sets cancel_requested_ only; done() stays false
//                              until notifyDone() is called externally
// This is the KEY difference from MockIKVCacheRecvTask which makes cancel()
// always set done_=true, hiding the race condition bug.
// ---------------------------------------------------------------------------
class InflightMockRecvTask: public transfer::IKVCacheRecvTask {
public:
    bool done() const override {
        return done_.load(std::memory_order_acquire);
    }
    bool success() const override {
        return success_.load(std::memory_order_relaxed);
    }

    void cancel() override {
        cancel_requested_.store(true, std::memory_order_release);
        if (!transferring_.load(std::memory_order_acquire)) {
            success_.store(false, std::memory_order_relaxed);
            error_code_ = transfer::TransferErrorCode::CANCELLED;
            done_.store(true, std::memory_order_release);
        }
    }

    void forceCancel() override {
        cancel_requested_.store(true, std::memory_order_release);
        success_.store(false, std::memory_order_relaxed);
        error_code_ = transfer::TransferErrorCode::CANCELLED;
        done_.store(true, std::memory_order_release);
    }

    transfer::TransferErrorCode errorCode() const override {
        return error_code_;
    }
    std::string errorMessage() const override {
        return success_.load() ? "" : "mock inflight task failed/cancelled";
    }

    void startTransferring() {
        transferring_.store(true, std::memory_order_release);
    }

    void notifyDone(bool ok) {
        success_.store(ok, std::memory_order_relaxed);
        if (!ok) {
            error_code_ = transfer::TransferErrorCode::UNKNOWN;
        }
        done_.store(true, std::memory_order_release);
    }

    bool isCancelRequested() const {
        return cancel_requested_.load(std::memory_order_acquire);
    }
    bool isTransferring() const {
        return transferring_.load(std::memory_order_acquire);
    }

private:
    std::atomic<bool>           done_{false};
    std::atomic<bool>           success_{true};
    std::atomic<bool>           cancel_requested_{false};
    std::atomic<bool>           transferring_{false};
    transfer::TransferErrorCode error_code_{transfer::TransferErrorCode::OK};
};

// ---------------------------------------------------------------------------
// InflightMockReceiver — creates InflightMockRecvTask instances
// ---------------------------------------------------------------------------
class InflightMockReceiver: public transfer::IKVCacheReceiver {
public:
    bool regMem(const BlockInfo&, uint64_t = 0) override {
        return true;
    }

    transfer::IKVCacheRecvTaskPtr recv(const transfer::RecvRequest& request) override {
        auto                        task = std::make_shared<InflightMockRecvTask>();
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_[request.unique_key] = task;
        return task;
    }

    void stealTask(const std::string&) override {
        steal_count_.fetch_add(1, std::memory_order_relaxed);
    }

    transfer::IKVCacheRecvTaskPtr getTask(const std::string& unique_key) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = tasks_.find(unique_key);
        return it != tasks_.end() ? it->second : nullptr;
    }

    std::shared_ptr<InflightMockRecvTask> getInflightTask(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = tasks_.find(key);
        return it != tasks_.end() ? it->second : nullptr;
    }

    int taskCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(tasks_.size());
    }

    bool hasEnoughTasks(const std::string& base_key, int expected) const {
        std::lock_guard<std::mutex> lock(mutex_);
        int                         count = 0;
        for (const auto& [k, _] : tasks_) {
            if (k.size() > base_key.size() && k.substr(0, base_key.size()) == base_key && k[base_key.size()] == '_')
                ++count;
        }
        return count >= expected;
    }

private:
    mutable std::mutex                                                     mutex_;
    std::unordered_map<std::string, std::shared_ptr<InflightMockRecvTask>> tasks_;
    std::atomic<int>                                                       steal_count_{0};
};

// ---------------------------------------------------------------------------
// MockLayerBlockConverter — trivial stub
// ---------------------------------------------------------------------------
class LeaseTestLayerBlockConverter: public LayerBlockConverter {
public:
    std::vector<BlockInfo> convertIndexToBuffer(int, int, int, int) const override {
        return {};
    }
    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        return {};
    }
};

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class DecodeLeaseRaceTest: public ::testing::Test {
protected:
    void SetUp() override {
        config_.transfer_backend_config.cache_store_rdma_mode        = false;
        config_.transfer_backend_config.messager_io_thread_count     = 1;
        config_.transfer_backend_config.messager_worker_thread_count = 1;
        config_.tp_size                                              = 1;
        config_.tp_rank                                              = 0;
        config_.layer_all_num                                        = 2;
        config_.transfer_backend_config.cache_store_listen_port      = 0;

        converter_ = std::make_shared<LeaseTestLayerBlockConverter>();
        receiver_  = std::make_shared<InflightMockReceiver>();
        decode_    = std::make_unique<P2PConnectorWorkerDecode>(config_, converter_, nullptr, receiver_);
    }

    std::shared_ptr<LayerCacheBuffer> makeBuffer(int layer_id, int blocks = 2) {
        auto buf = std::make_shared<LayerCacheBuffer>(layer_id);
        for (int i = 0; i < blocks; ++i)
            buf->addBlockId(layer_id * 1000 + i, i);
        return buf;
    }

    std::vector<std::shared_ptr<LayerCacheBuffer>> makeBuffers(int layers = 2, int blocks = 2) {
        std::vector<std::shared_ptr<LayerCacheBuffer>> v;
        for (int i = 0; i < layers; ++i)
            v.push_back(makeBuffer(i, blocks));
        return v;
    }

    std::string layerKey(const std::string& base, int layer, int partition = 0) {
        return base + "_" + std::to_string(layer) + "_" + std::to_string(partition);
    }

    void waitForTasks(const std::string& base, int n, int timeout_ms = 500) {
        for (int i = 0; i < timeout_ms; ++i) {
            if (receiver_->hasEnoughTasks(base, n))
                return;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    P2PConnectorWorkerConfig                      config_;
    std::shared_ptr<LeaseTestLayerBlockConverter> converter_;
    std::shared_ptr<InflightMockReceiver>         receiver_;
    std::unique_ptr<P2PConnectorWorkerDecode>     decode_;
};

// ===========================================================================
// DecodeTargetWriteLease standalone tests
// ===========================================================================

TEST(DecodeTargetWriteLeaseTest, InitialState) {
    DecodeTargetWriteLease lease;
    EXPECT_FALSE(lease.isSealed());
    EXPECT_FALSE(lease.isStopped());
    EXPECT_EQ(lease.startedOps(), 0);
    EXPECT_EQ(lease.finishedOps(), 0);
}

TEST(DecodeTargetWriteLeaseTest, SealAlone_NoOps_IsStopped) {
    DecodeTargetWriteLease lease;
    lease.seal();
    EXPECT_TRUE(lease.isSealed());
    EXPECT_TRUE(lease.isStopped());
}

TEST(DecodeTargetWriteLeaseTest, SealWithPendingOps_NotStopped) {
    DecodeTargetWriteLease lease;
    lease.onTransferStarted();
    lease.onTransferStarted();
    lease.seal();
    EXPECT_TRUE(lease.isSealed());
    EXPECT_FALSE(lease.isStopped());
    EXPECT_EQ(lease.startedOps(), 2);
    EXPECT_EQ(lease.finishedOps(), 0);
}

TEST(DecodeTargetWriteLeaseTest, AllOpsFinished_AfterSeal_Stopped) {
    DecodeTargetWriteLease lease;
    lease.onTransferStarted();
    lease.onTransferStarted();
    lease.seal();
    lease.onTransferFinished();
    EXPECT_FALSE(lease.isStopped());
    lease.onTransferFinished();
    EXPECT_TRUE(lease.isStopped());
}

TEST(DecodeTargetWriteLeaseTest, NotSealed_AllOpsFinished_NotStopped) {
    DecodeTargetWriteLease lease;
    lease.onTransferStarted();
    lease.onTransferFinished();
    EXPECT_FALSE(lease.isStopped());
}

TEST(DecodeTargetWriteLeaseTest, ConcurrentStartFinish) {
    DecodeTargetWriteLease   lease;
    const int                N = 100;
    std::vector<std::thread> threads;
    for (int i = 0; i < N; ++i) {
        threads.emplace_back([&lease]() { lease.onTransferStarted(); });
    }
    for (auto& t : threads)
        t.join();
    EXPECT_EQ(lease.startedOps(), N);

    threads.clear();
    for (int i = 0; i < N; ++i) {
        threads.emplace_back([&lease]() { lease.onTransferFinished(); });
    }
    for (auto& t : threads)
        t.join();
    EXPECT_EQ(lease.finishedOps(), N);

    lease.seal();
    EXPECT_TRUE(lease.isStopped());
}

// ===========================================================================
// Integration tests: cancel + lease lifecycle in P2PConnectorWorkerDecode
// ===========================================================================

// 1. Normal completion: all tasks done → lease erased, queryLeaseStatus returns false
TEST_F(DecodeLeaseRaceTest, NormalCompletion_LeaseErased) {
    std::string key      = "lease_normal_ok";
    auto        bufs     = makeBuffers();
    int64_t     deadline = currentTimeMs() + 5000;

    std::thread reader([&]() { decode_->read(1, key, deadline, bufs); });

    waitForTasks(key, 2);
    for (int i = 0; i < 2; ++i) {
        auto t = receiver_->getInflightTask(layerKey(key, i));
        ASSERT_NE(t, nullptr);
        t->startTransferring();
        t->notifyDone(true);
    }
    reader.join();

    bool sealed, stopped;
    int  started, finished;
    bool found = decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_FALSE(found);
    EXPECT_TRUE(stopped);
}

// 2. Cancel with PENDING tasks (not yet transferring) → tasks done immediately,
//    lease sealed+stopped, entry erased on query
TEST_F(DecodeLeaseRaceTest, CancelPendingTasks_LeaseStoppedImmediately) {
    std::string key      = "lease_cancel_pending";
    auto        bufs     = makeBuffers();
    int64_t     deadline = currentTimeMs() + 5000;

    std::atomic<bool> read_done{false};
    ErrorInfo         result;
    std::thread       reader([&]() {
        result    = decode_->read(1, key, deadline, bufs);
        read_done = true;
    });

    waitForTasks(key, 2);

    bool cancelled = false;
    for (int i = 0; i < 50 && !cancelled; ++i) {
        cancelled = decode_->cancelRead(key);
        if (!cancelled)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(cancelled);

    for (int w = 0; w < 200 && !read_done; ++w)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    reader.join();

    EXPECT_TRUE(result.hasError());
    EXPECT_EQ(result.code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED);

    bool sealed, stopped;
    int  started, finished;
    bool found = decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_TRUE(sealed);
    EXPECT_TRUE(stopped);
}

// 3. KEY TEST: Cancel while tasks are TRANSFERRING → lease stays alive until
//    all in-flight transfers complete. queryLeaseStatus must NOT report stopped
//    while transfers are in-flight.
TEST_F(DecodeLeaseRaceTest, CancelInflight_LeaseStaysUntilTransfersDone) {
    std::string key      = "lease_cancel_inflight";
    auto        bufs     = makeBuffers();
    int64_t     deadline = currentTimeMs() + 5000;

    std::atomic<bool> read_done{false};
    ErrorInfo         result;
    std::thread       reader([&]() {
        result    = decode_->read(1, key, deadline, bufs);
        read_done = true;
    });

    waitForTasks(key, 2);

    auto task0 = receiver_->getInflightTask(layerKey(key, 0));
    auto task1 = receiver_->getInflightTask(layerKey(key, 1));
    ASSERT_NE(task0, nullptr);
    ASSERT_NE(task1, nullptr);
    task0->startTransferring();
    task1->startTransferring();

    bool cancelled = false;
    for (int i = 0; i < 50 && !cancelled; ++i) {
        cancelled = decode_->cancelRead(key);
        if (!cancelled)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(cancelled);

    EXPECT_FALSE(task0->done());
    EXPECT_FALSE(task1->done());

    for (int w = 0; w < 500 && !read_done; ++w)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // read() should NOT have returned yet because InflightMock tasks are still "not done"
    // Actually — read() returns on Cancelled path once cancelled flag is set,
    // regardless of task done() status. The key invariant is about the LEASE.
    // read() has returned, but lease_map_ entry must still be alive.

    bool sealed, stopped;
    int  started, finished;
    bool found = decode_->queryLeaseStatus(key, sealed, started, finished, stopped);

    // THIS IS THE CRITICAL ASSERTION:
    // Before the fix, lease_map_ was erased in read() on Cancelled path,
    // so found would be false and stopped would be true → premature block release.
    // After the fix, the entry stays alive until all transfers complete.
    EXPECT_TRUE(found) << "BUG: lease erased while transfers still in-flight";
    EXPECT_TRUE(sealed);
    EXPECT_EQ(started, 2);
    EXPECT_FALSE(stopped) << "BUG: lease reports stopped while transfers still in-flight";

    // Now simulate transport layer completing the transfers
    task0->notifyDone(false);

    found = decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_TRUE(found);
    EXPECT_EQ(finished, 1);
    EXPECT_FALSE(stopped);

    task1->notifyDone(false);

    found = decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    // After all transfers done, queryLeaseStatus erases the entry and returns true
    // with stopped=true, OR it might already be erased. Either way, stopped must be true.
    EXPECT_TRUE(stopped) << "lease should be stopped after all transfers completed";

    // Subsequent query: entry should be gone
    found = decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_FALSE(found);
    EXPECT_TRUE(stopped);

    reader.join();
}

// 4. Cancel with mixed PENDING and TRANSFERRING tasks
TEST_F(DecodeLeaseRaceTest, CancelMixed_PendingAndInflight) {
    std::string key      = "lease_cancel_mixed";
    auto        bufs     = makeBuffers();
    int64_t     deadline = currentTimeMs() + 5000;

    std::atomic<bool> read_done{false};
    ErrorInfo         result;
    std::thread       reader([&]() {
        result    = decode_->read(1, key, deadline, bufs);
        read_done = true;
    });

    waitForTasks(key, 2);

    auto task0 = receiver_->getInflightTask(layerKey(key, 0));
    auto task1 = receiver_->getInflightTask(layerKey(key, 1));
    ASSERT_NE(task0, nullptr);
    ASSERT_NE(task1, nullptr);

    // task0 is TRANSFERRING, task1 stays PENDING
    task0->startTransferring();

    bool cancelled = false;
    for (int i = 0; i < 50 && !cancelled; ++i) {
        cancelled = decode_->cancelRead(key);
        if (!cancelled)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(cancelled);

    // task1 (PENDING) should be done immediately after cancel
    EXPECT_TRUE(task1->done());
    // task0 (TRANSFERRING) should NOT be done
    EXPECT_FALSE(task0->done());

    for (int w = 0; w < 200 && !read_done; ++w)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

    bool sealed, stopped;
    int  started, finished;
    bool found = decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_TRUE(found);
    EXPECT_TRUE(sealed);
    // task1 is done (cancelled immediately), but task0 is still in-flight
    EXPECT_FALSE(stopped);

    // Complete task0
    task0->notifyDone(false);

    found = decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_TRUE(stopped);

    reader.join();
}

// 5. ReturnDeadlineIncomplete: timeout before all tasks done, lease stays for cleanup
TEST_F(DecodeLeaseRaceTest, ReturnDeadline_LeaseStaysForCleanup) {
    std::string key      = "lease_return_deadline";
    auto        bufs     = makeBuffers();
    int64_t     deadline = currentTimeMs() + 20;  // very short deadline

    std::atomic<bool> read_done{false};
    ErrorInfo         result;
    std::thread       reader([&]() {
        result    = decode_->read(1, key, deadline, bufs);
        read_done = true;
    });

    waitForTasks(key, 2);

    auto task0 = receiver_->getInflightTask(layerKey(key, 0));
    auto task1 = receiver_->getInflightTask(layerKey(key, 1));
    if (task0)
        task0->startTransferring();
    if (task1)
        task1->startTransferring();

    for (int w = 0; w < 500 && !read_done; ++w)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    reader.join();

    EXPECT_TRUE(result.hasError());
    EXPECT_EQ(result.code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE);

    bool sealed, stopped;
    int  started, finished;
    bool found = decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_TRUE(found);
    EXPECT_TRUE(sealed);
    EXPECT_FALSE(stopped);

    if (task0 && !task0->done())
        task0->notifyDone(false);
    if (task1 && !task1->done())
        task1->notifyDone(false);

    found = decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_TRUE(stopped);
}

// 6. queryLeaseStatus on unknown key returns false + stopped=true
TEST_F(DecodeLeaseRaceTest, QueryUnknownKey_ReturnsFalse) {
    bool sealed, stopped;
    int  started, finished;
    bool found = decode_->queryLeaseStatus("nonexistent_key", sealed, started, finished, stopped);
    EXPECT_FALSE(found);
    EXPECT_TRUE(stopped);
    EXPECT_TRUE(sealed);
    EXPECT_EQ(started, 0);
    EXPECT_EQ(finished, 0);
}

// 7. Multiple cancel calls are idempotent
TEST_F(DecodeLeaseRaceTest, DoubleCancelIsIdempotent) {
    std::string key      = "lease_double_cancel";
    auto        bufs     = makeBuffers();
    int64_t     deadline = currentTimeMs() + 5000;

    std::atomic<bool> read_done{false};
    ErrorInfo         result;
    std::thread       reader([&]() {
        result    = decode_->read(1, key, deadline, bufs);
        read_done = true;
    });

    waitForTasks(key, 2);

    auto task0 = receiver_->getInflightTask(layerKey(key, 0));
    auto task1 = receiver_->getInflightTask(layerKey(key, 1));
    task0->startTransferring();
    task1->startTransferring();

    bool c1 = false;
    for (int i = 0; i < 50 && !c1; ++i) {
        c1 = decode_->cancelRead(key);
        if (!c1)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(c1);

    for (int w = 0; w < 200 && !read_done; ++w)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // read() has returned, so read_tasks_ is erased — second cancel returns false
    bool c2 = decode_->cancelRead(key);
    EXPECT_FALSE(c2);

    task0->notifyDone(false);
    task1->notifyDone(false);

    bool sealed, stopped;
    int  started, finished;
    decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_TRUE(stopped);

    reader.join();
}

// 8. Lease started_ops matches number of recv tasks created
TEST_F(DecodeLeaseRaceTest, LeaseStartedOps_MatchesTaskCount) {
    std::string key      = "lease_started_ops_count";
    auto        bufs     = makeBuffers(3, 1);  // 3 layers, 1 block each
    int64_t     deadline = currentTimeMs() + 5000;

    std::atomic<bool> read_done{false};
    std::thread       reader([&]() {
        decode_->read(1, key, deadline, bufs);
        read_done = true;
    });

    waitForTasks(key, 3);

    bool sealed, stopped;
    int  started, finished;
    bool found = decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_TRUE(found);
    EXPECT_EQ(started, 3);
    EXPECT_EQ(finished, 0);

    for (int i = 0; i < 3; ++i) {
        auto t = receiver_->getInflightTask(layerKey(key, i));
        if (t) {
            t->startTransferring();
            t->notifyDone(true);
        }
    }

    for (int w = 0; w < 200 && !read_done; ++w)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    reader.join();
}

// 9. Gradual finish: queryLeaseStatus tracks incremental progress
TEST_F(DecodeLeaseRaceTest, GradualFinish_IncrementalProgress) {
    std::string key      = "lease_gradual";
    auto        bufs     = makeBuffers(3, 1);
    int64_t     deadline = currentTimeMs() + 5000;

    std::atomic<bool> read_done{false};
    ErrorInfo         result;
    std::thread       reader([&]() {
        result    = decode_->read(1, key, deadline, bufs);
        read_done = true;
    });

    waitForTasks(key, 3);

    // Start all tasks as transferring
    std::vector<std::shared_ptr<InflightMockRecvTask>> tasks;
    for (int i = 0; i < 3; ++i) {
        auto t = receiver_->getInflightTask(layerKey(key, i));
        ASSERT_NE(t, nullptr);
        t->startTransferring();
        tasks.push_back(t);
    }

    // Cancel while all are in-flight
    bool cancelled = false;
    for (int i = 0; i < 50 && !cancelled; ++i) {
        cancelled = decode_->cancelRead(key);
        if (!cancelled)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(cancelled);

    for (int w = 0; w < 200 && !read_done; ++w)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

    bool sealed, stopped;
    int  started, finished;

    // Finish tasks one by one and verify incremental progress
    tasks[0]->notifyDone(false);
    decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_EQ(finished, 1);
    EXPECT_FALSE(stopped);

    tasks[1]->notifyDone(false);
    decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_EQ(finished, 2);
    EXPECT_FALSE(stopped);

    tasks[2]->notifyDone(false);
    decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_EQ(finished, 3);
    EXPECT_TRUE(stopped);

    reader.join();
}

// 10. Stress test: many concurrent cancel+query cycles
TEST_F(DecodeLeaseRaceTest, StressConcurrentCancelAndQuery) {
    std::string key      = "lease_stress";
    auto        bufs     = makeBuffers(4, 1);
    int64_t     deadline = currentTimeMs() + 5000;

    std::atomic<bool> read_done{false};
    std::thread       reader([&]() {
        decode_->read(1, key, deadline, bufs);
        read_done = true;
    });

    waitForTasks(key, 4);

    std::vector<std::shared_ptr<InflightMockRecvTask>> tasks;
    for (int i = 0; i < 4; ++i) {
        auto t = receiver_->getInflightTask(layerKey(key, i));
        ASSERT_NE(t, nullptr);
        t->startTransferring();
        tasks.push_back(t);
    }

    decode_->cancelRead(key);

    for (int w = 0; w < 200 && !read_done; ++w)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // Concurrent queries while tasks finish gradually
    std::atomic<bool> query_stop{false};
    std::thread       querier([&]() {
        while (!query_stop.load()) {
            bool sealed, stopped;
            int  started, finished;
            decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    for (int i = 0; i < 4; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        tasks[i]->notifyDone(false);
    }

    // Final check: must be stopped
    bool sealed, stopped;
    int  started, finished;
    // Give querier a chance to drain
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    query_stop = true;
    querier.join();

    decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_TRUE(stopped);

    reader.join();
}

// 11. AllDone path erases lease immediately — no leak
TEST_F(DecodeLeaseRaceTest, AllDone_LeaseErasedImmediately) {
    std::string key      = "lease_alldone_erase";
    auto        bufs     = makeBuffers(1, 1);
    int64_t     deadline = currentTimeMs() + 5000;

    std::thread reader([&]() { decode_->read(1, key, deadline, bufs); });

    waitForTasks(key, 1);
    auto task = receiver_->getInflightTask(layerKey(key, 0));
    ASSERT_NE(task, nullptr);
    task->startTransferring();
    task->notifyDone(true);

    reader.join();

    bool sealed, stopped;
    int  started, finished;
    bool found = decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_FALSE(found);
    EXPECT_TRUE(stopped);
}

// 12. Cancel-after-read-returned: cancelRead returns false, lease still queryable
TEST_F(DecodeLeaseRaceTest, CancelAfterReadReturned_Inflight) {
    std::string key      = "lease_cancel_after_return";
    auto        bufs     = makeBuffers();
    int64_t     deadline = currentTimeMs() + 20;

    std::atomic<bool> read_done{false};
    std::thread       reader([&]() {
        decode_->read(1, key, deadline, bufs);
        read_done = true;
    });

    waitForTasks(key, 2);
    auto task0 = receiver_->getInflightTask(layerKey(key, 0));
    auto task1 = receiver_->getInflightTask(layerKey(key, 1));
    if (task0)
        task0->startTransferring();
    if (task1)
        task1->startTransferring();

    for (int w = 0; w < 500 && !read_done; ++w)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    reader.join();

    // read() already returned (ReturnDeadlineIncomplete), read_tasks_ erased
    bool c = decode_->cancelRead(key);
    EXPECT_FALSE(c);

    // But lease_map_ entry should still be alive
    bool sealed, stopped;
    int  started, finished;
    bool found = decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_TRUE(found);
    EXPECT_FALSE(stopped);

    if (task0)
        task0->notifyDone(false);
    if (task1)
        task1->notifyDone(false);

    decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
    EXPECT_TRUE(stopped);
}

}  // namespace test
}  // namespace rtp_llm
