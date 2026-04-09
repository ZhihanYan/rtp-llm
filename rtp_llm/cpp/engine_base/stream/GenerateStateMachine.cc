#include "rtp_llm/cpp/engine_base/stream/GenerateStateMachine.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"

using namespace std;

namespace rtp_llm {
// ============================================================================
// GenerateStateMachine method implementations
// ============================================================================

StreamState GenerateStateMachine::moveToNext() {
    // Error 最高优先级，任何状态下直接终止
    if (events_.has(StreamEvents::Error)) {
        status = StreamState::FINISHED;
        releaseResource();
        return status;
    }

    switch (status) {
        case StreamState::WAITING:
            handleWaiting();
            break;
        case StreamState::LOADING_CACHE:
            handleLoading();
            break;
        case StreamState::RUNNING:
            handleRunning();
            break;
        case StreamState::FINISHED:
            break;
        default:
            RTP_LLM_LOG_ERROR("Error: Unrecognized Generate State");
            if (error_info.ok()) {
                error_info = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "Error: Unrecognized Generate State");
            }
            status = StreamState::FINISHED;
            releaseResource();
            break;
    }
    return status;
}

void GenerateStateMachine::handleWaiting() {
    if (!events_.has(StreamEvents::CanRun)) {
        return;
    }
    // LoadInitiated 未设置时，必须先执行 initKVBlock 和 asyncLoadCache
    if (!events_.has(StreamEvents::LoadInitiated)) {
        auto result = stream_cache_resource_->initKVBlock(reserve_step_);
        if (!result.ok()) {
            error_info = ErrorInfo(ErrorCode::MALLOC_FAILED, "LACK MEM");
            status = StreamState::FINISHED;
            releaseResource();
            return;
        }
        bool ret = stream_cache_resource_->asyncLoadCache();
        // 设置 LoadInitiated 标志，表示已尝试asyncLoadCache. 当前实现即便asyncLoadCache失败也不再重试
        reportEvent(StreamEvents::LoadInitiated);
        if (ret) {
            status = StreamState::LOADING_CACHE;
        }
        return;
    }
    auto result = stream_cache_resource_->incrKVBlock(reserve_step_);
    if (!result.ok()) {
        error_info = ErrorInfo(ErrorCode::MALLOC_FAILED, "LACK MEM");
        status = StreamState::FINISHED;
        releaseResource();
        return;
    }
    status = StreamState::RUNNING;
    return;
}

void GenerateStateMachine::handleLoading() {
    if (stream_cache_resource_->loadCacheDone()) {
        status = StreamState::WAITING;
    }
}

void GenerateStateMachine::handleRunning() {
    // in pd sep case，kvcache could be released after remote load done.
    if (events_.has(StreamEvents::GenerateDone)) {
        status = StreamState::FINISHED;
        releaseResource();
        return;
    }
    auto result = stream_cache_resource_->incrKVBlock(reserve_step_);
    if (!result.ok()) {
        // Report Error event so moveToNext() won't be called again on this stream
        reportEvent(StreamEvents::Error, ErrorCode::MALLOC_FAILED, "incrKVBlock failed: LACK MEM");
        status = StreamState::FINISHED;
        releaseResource();
    }
}

void GenerateStateMachine::releaseResource() {
    if (!stream_cache_resource_->isResourceReleased()) {
        stream_cache_resource_->releaseResource();
    }
}
}