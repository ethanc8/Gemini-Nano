# tflite flatbuffer schemas
import tflite.Model, tflite.SubGraph, tflite.TensorType
import sys

import safetensors.torch as st

import torch

# from mediapipe.tasks.cc.genai.inference.proto import llm_params_pb2

# FIXME: sentencepiece and mediapipe protobuf conflict due to both setting some abseil flag
# import sentencepiece as spm

input_file = open(sys.argv[1], "rb")
buf = bytearray(input_file.read())

model: tflite.Model.Model = tflite.Model.Model.GetRootAs(buf)

# print(f"tflite file version: {model.Version()}")

graph: tflite.SubGraph.SubGraph = model.Subgraphs(0)

# print(f"Model name: {graph.Name()}")

# print("===METADATA===")

# parameters: llm_params_pb2.LlmParameters = None

# for i in range(model.MetadataLength()):
#     metametadata = model.Metadata(i)
#     metadata_buf = model.Buffers(metametadata.Buffer())
#     metadata = buf[metadata_buf.Offset():metadata_buf.Offset()+metadata_buf.Size()]
#     print(metametadata.Name().decode("utf-8"))
#     if metadata_buf.Size() < 80: # Don't print huge buffers
#         print(metadata)
#     if metametadata.Name() == b'odml.infra.proto.LlmParameters':
#         parameters = llm_params_pb2.LlmParameters()
#         parameters.ParseFromString(metadata)
#         print(parameters)
#     elif metametadata.Name() == b'spm_vocab_model':
#         # We can load it with sentencepiece like so:
#         # sp = spm.SentencePieceProcessor()
#         # sp.load_from_serialized_proto(metadata)
#         pass
#     elif metametadata.Name() == b'odml.infra.LlmModelType':
#         print(metadata[0])

# # sys.exit(0)

print("===TENSORS===")

name_of_tensor_type = {
     0: "FLOAT32",
     9: "INT8   ",
    17: "INT4   ",
}

dtype_for_tensor_type = {
    0: torch.float32,
    9: torch.int8,
    17: torch.uint8 # because torch.int4 doesn't exist
}

# Reversed from https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/genai/converter/safetensors_converter.py#L433
def update_target_name(target_name: str) -> str:
    """Updates the target name to match the tensor name convention."""
    # target_name = reverse_replace(target_name, "base_model.model.", "")
    def reverse_replace(theStr: str, a, b):
        return theStr.replace(b, a)
    
    target_name = reverse_replace(target_name, ".weight", ".w")
    target_name = reverse_replace(target_name, 
        "model.layers.", "params.lm.transformer.x_layers_"
    )

    target_name = reverse_replace(target_name, 
        "mlp.gate_proj", "ff_layer.ffn_layer1_gate"
    )
    target_name = reverse_replace(target_name, "mlp.up_proj", "ff_layer.ffn_layer1")
    target_name = reverse_replace(target_name, "mlp.down_proj", "ff_layer.ffn_layer2")

    # The StableLM converter uses this post_attention_layernorm for the "cpu" backend.
    # target_name = reverse_replace(target_name, 
    #     "ff_layer.pre_layer_norm.weight", "ff_layer.pre_layer_norm.scale"
    # )
    # target_name = reverse_replace(target_name, 
    #     "post_attention_layernorm", "ff_layer.pre_layer_norm"
    # )
    # The StableLM converter uses this post_attention_layernorm for the "gpu" backend.
    # We have both post_attention_layernorms, which is a bit concerning.
    # We assume the one for the "gpu" backend is used, because the model was
    # compiled for the "gpu" backend.
    target_name = reverse_replace(target_name,
        "post_layer_norm.weight", "post_layer_norm.scale"
    )
    target_name = reverse_replace(target_name,
        "post_attention_layernorm", "post_layer_norm"
    )
    
    target_name = reverse_replace(target_name, 
        "pre_layer_norm.weight", "pre_layer_norm.scale"
    )
    target_name = reverse_replace(target_name, "input_layernorm", "pre_layer_norm")
    
    target_name = reverse_replace(target_name, "self_attn.q_proj", "self_attention.q")
    target_name = reverse_replace(target_name, "self_attn.k_proj", "self_attention.k")
    target_name = reverse_replace(target_name, "self_attn.v_proj", "self_attention.v")
    target_name = reverse_replace(target_name, "self_attn.o_proj", "self_attention.post")
    target_name = reverse_replace(target_name, 
        "model.embed_tokens", "params.lm.softmax.logits_ffn"
    )
    target_name = reverse_replace(target_name, "final_ln.weight", "final_ln.scale")
    target_name = reverse_replace(target_name, "model.norm", "params.lm.final_ln")
    
    

    # # For LoRA weights
    # if "post" in target_name:
    #   target_name = reverse_replace(target_name, "lora_A.w", "w_prime_right")
    #   target_name = reverse_replace(target_name, "lora_B.w", "w_prime_left")
    # else:
    #   target_name = reverse_replace(target_name, "lora_A.w", "w_prime_left")
    #   target_name = reverse_replace(target_name, "lora_B.w", "w_prime_right")

    return target_name

tensor_dict = {}

for i in range(graph.TensorsLength()):
    tensor = graph.Tensors(i)
    tensor_size = tensor.Shape(0)
    tensor_name = tensor.Name().decode("utf-8")
    tensor_type: tflite.TensorType = tensor.Type()
    tensor_buf = model.Buffers(tensor.Buffer())
    target_name = update_target_name(tensor_name)

    print(f"{name_of_tensor_type[tensor_type]} {tensor_size} {tensor_name}")
    
    shape = None
    if(("self_attn.q_proj.weight" in target_name
     or "self_attn.o_proj.weight" in target_name
       ) and tensor_size == 4194304):
        shape = (8192, 512)
    elif(("self_attn.k_proj.weight" in target_name
       or "self_attn.v_proj.weight" in target_name
         ) and tensor_size == 524288):
        shape = (1024, 512)
    elif(("mlp.up_proj" in target_name
       or "mlp.gate_proj" in target_name
         ) and tensor_size == 12582912):
        shape = (49152, 256)
    elif(("mlp.down_proj" in target_name
         ) and tensor_size == 12582912):
        shape = (8192, 1536)
    elif("model.embed_tokens.weight" == target_name
           and tensor_size == 262275072):
        shape = (1024512, 256)
    else:
        # shape = (tensor_size,)
        pass
    
    print(f">> {shape} {target_name}")
    # ({tensor_buf.Size()}, {tensor_buf.Size() / tensor_size} B/element)
    # it's 1 B/element
    
    tensor_data = torch.frombuffer(buffer=buf, 
                                   dtype=dtype_for_tensor_type[tensor_type], 
                                   offset=tensor_buf.Offset(),
                                   count=tensor_buf.Size(),
                                  )
    
    if shape is not None:
        tensor_data.reshape(shape)

    tensor_dict[target_name] = tensor_data

st.save_file(tensor_dict, sys.argv[2])
print(f"Success! Saved to {sys.argv[2]}")

