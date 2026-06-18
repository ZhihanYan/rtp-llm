#include <memory>

#include "gtest/gtest.h"
#include "torch/all.h"

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {

class BatchDecodeSchedulerTest: public DeviceTestBase {
protected:
    GenerateStreamPtr createStream() {
        auto generate_input             = std::make_shared<GenerateInput>();
        auto generate_config            = std::make_shared<GenerateConfig>();
        generate_input->generate_config = generate_config;
        generate_input->input_ids       = torch::tensor({1}, torch::kInt32);
        ModelConfig     model_config;
        RuntimeConfig   runtime_config;
        ResourceContext resource_context;
        model_config.max_seq_len = 2048;
        return std::make_shared<NormalGenerateStream>(
            generate_input, model_config, runtime_config, resource_context, nullptr);
    }
};

TEST_F(BatchDecodeSchedulerTest, FinishedCandidateIsRemovedBeforeFullBatch) {
    RuntimeConfig runtime_config;
    runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 2;
    BatchDecodeScheduler scheduler(runtime_config, nullptr, nullptr);

    auto stream                      = createStream();
    stream->generate_status_->status = StreamState::FINISHED;

    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    ASSERT_EQ(scheduler.onflightStreams(), 1);

    auto scheduled = scheduler.schedule();
    ASSERT_TRUE(scheduled.ok());
    EXPECT_TRUE(scheduled.value().empty());
    EXPECT_EQ(scheduler.onflightStreams(), 0);
}

}  // namespace rtp_llm
