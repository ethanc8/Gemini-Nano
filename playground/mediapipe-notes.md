# Notes on how MediaPipe loads LLMs

## Getting the LlmParameters

[`LlmInferenceEngine_CreateSession_Helper`](https://github.com/google-ai-edge/mediapipe/blob/24a5b2d72babfa98d79664e7d2b52ebf7a0a7cbc/mediapipe/tasks/cc/genai/inference/c/llm_inference_engine_cpu.cc#L161) creates a [`ModelData` structure](https://github.com/google-ai-edge/mediapipe/blob/24a5b2d72babfa98d79664e7d2b52ebf7a0a7cbc/mediapipe/tasks/cc/genai/inference/utils/llm_utils/model_data.cc), which reads the parameters metadata.

This metadata is a protobuf of type [`odml.infra.proto.LlmParameters`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/genai/inference/proto/llm_params.proto#L54) stored in the metadata field `odml.infra.proto.LlmParameters`. [`ModelData::InitLlmParameters`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/genai/inference/utils/llm_utils/model_data.cc#L146)