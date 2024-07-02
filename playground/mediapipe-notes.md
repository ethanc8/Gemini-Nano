# Notes on how MediaPipe loads LLMs

## Getting the LlmParameters

[`LlmInferenceEngine_CreateSession_Helper`](https://github.com/google-ai-edge/mediapipe/blob/24a5b2d72babfa98d79664e7d2b52ebf7a0a7cbc/mediapipe/tasks/cc/genai/inference/c/llm_inference_engine_cpu.cc#L161) creates a [`model_data` structure](https://github.com/google-ai-edge/mediapipe/blob/24a5b2d72babfa98d79664e7d2b52ebf7a0a7cbc/mediapipe/tasks/cc/genai/inference/utils/llm_utils/model_data.cc), which reads the parameters metadata.

This metadata is a C string (`const char*`) in the metadata field `odml.infra.proto.LlmParameters`. 
