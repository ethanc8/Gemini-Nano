# Protobuf notes

[Protobuf v25 docs](https://protobuf-25-docs.github.io/)

> **Note:** We can just import the MediaPipe protobufs from MediaPipe, so this is not really necessary.

## Download the schemas

```bash
wget https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/tasks/cc/genai/inference/proto/llm_params.proto
wget https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/tasks/cc/genai/inference/proto/prompt_template.proto
wget https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/tasks/cc/genai/inference/proto/transformer_params.proto
```

## Converting the schema to Python boilerplate

```bash
mkdir protos
protoc --proto-path=buffer_schemas --python-out=protos buffer_schemas/mediapipe/tasks/cc/genai/inference/proto/llm_params.proto
```