# This is an optimized version of converter.py which was suggested by GPT-4o
import tflite.Model, tflite.SubGraph
from tflite.TensorType import TensorType
import sys
import torch
import safetensors.torch as st
import numpy as np
from multiprocessing import Pool

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

# Function to dequantize int4 tensor
def convert_quantized_int4_to_fp(quantized_data, scale_data, dims, dim_scale, dtype):
    zero_point = 8
    scaled_data = torch.zeros(dims[0] * dims[1], dtype=dtype)
    index = 0

    for i in range(dims[0]):
        for j in range(dims[1] // 2):
            scale = scale_data[j * 2] if dim_scale else scale_data[i]
            scaled_data[index] = (int(quantized_data[index // 2] & 0x0f) - zero_point) * scale
            index += 1
            scale = scale_data[j * 2 + 1] if dim_scale else scale_data[i]
            scaled_data[index] = (int(quantized_data[index // 2] >> 4) - zero_point) * scale
            index += 1

    return scaled_data

# Function to dequantize int8 tensor
def convert_quantized_int8_to_fp(quantized_data, scale_data, dims, dim_scale, dtype):
    scaled_data = torch.zeros(dims[0] * dims[1], dtype=dtype)
    index = 0

    for i in range(dims[0]):
        for j in range(dims[1]):
            scale = scale_data[j] if dim_scale else scale_data[i]
            scaled_data[index] = quantized_data[index] * scale
            index += 1

    return scaled_data

# Function to process tensors
def process_tensor(args):
    tensor_name, quantized_tensor, buf, scale_tensors, tensor_dims, model, TARGET_DTYPE = args
    quantized_buf_meta = model.Buffers(quantized_tensor.Buffer())
    scale_tensor_name = tensor_name + "_quantized_scale"
    scale_buf_meta = model.Buffers(scale_tensors[scale_tensor_name].Buffer())
    dims = tensor_dims[tensor_name]

    if quantized_tensor.Type() == TensorType.INT4:
        quantized_buf = torch.frombuffer(buffer=buf, dtype=torch.uint8, offset=quantized_buf_meta.Offset(), count=quantized_buf_meta.Size())
        scale_buf = torch.frombuffer(buffer=buf, dtype=torch.float32, offset=scale_buf_meta.Offset(), count=scale_buf_meta.Size() // 4)
        tensor_data = convert_quantized_int4_to_fp(quantized_data=quantized_buf, scale_data=scale_buf, dims=dims, dim_scale=0, dtype=TARGET_DTYPE)
    elif quantized_tensor.Type() == TensorType.INT8:
        quantized_buf = torch.frombuffer(buffer=buf, dtype=torch.int8, offset=quantized_buf_meta.Offset(), count=quantized_buf_meta.Size())
        scale_buf = torch.frombuffer(buffer=buf, dtype=torch.float32, offset=scale_buf_meta.Offset(), count=scale_buf_meta.Size() // 4)
        tensor_data = convert_quantized_int8_to_fp(quantized_data=quantized_buf, scale_data=scale_buf, dims=dims, dim_scale=0, dtype=TARGET_DTYPE)
    else:
        raise ValueError("Unknown tensor type")

    return update_target_name(tensor_name), tensor_data

# Main function
def main():
    if len(sys.argv) >= 4:
        if sys.argv[3] == "fp32":
            TARGET_DTYPE = torch.float32
        elif sys.argv[3] == "fp16":
            TARGET_DTYPE = torch.float16
        elif sys.argv[3] == "bf16":
            TARGET_DTYPE = torch.bfloat16
    else:
        print("WARNING: TARGET_DTYPE not properly specified, assuming fp32")
        TARGET_DTYPE = torch.float32

    input_file = open(sys.argv[1], "rb")
    buf = bytearray(input_file.read())
    model: tflite.Model.Model = tflite.Model.Model.GetRootAs(buf)
    graph: tflite.SubGraph.SubGraph = model.Subgraphs(0)

    name_of_tensor_type = {
         0: "FLOAT32",
         9: "INT8",
        17: "INT4",
    }

    dtype_for_tensor_type = {
        0: torch.float32,
        9: torch.int8,
        17: torch.uint8
    }

    size_for_tensor_type = {
        0: 4,
        9: 1,
        17: 0.5
    }

    i4_tensors = {}
    i8_tensors = {}
    fp32_tensors = {}
    scale_tensors = {}
    tensor_dims = {}

    for i in range(graph.TensorsLength()):
        tensor = graph.Tensors(i)
        tensor_name = tensor.Name().decode("utf-8")
        tensor_type: TensorType = tensor.Type()

        if tensor_name.endswith(".w_quantized_scale"):
            scale_tensors[tensor_name] = tensor
        elif tensor_type == TensorType.INT4:
            i4_tensors[tensor_name] = tensor
        elif tensor_type == TensorType.INT8:
            i8_tensors[tensor_name] = tensor
        elif tensor_type == TensorType.FLOAT32:
            fp32_tensors[tensor_name] = tensor

        tensor_buf_size = tensor.Shape(0)
        tensor_size = tensor_buf_size // size_for_tensor_type[tensor_type]

        shape = None
        if (".self_attention.q." in tensor_name or ".self_attention.post." in tensor_name) and tensor_size == 4194304:
            shape = (2048, 2048)
        elif (".self_attention.k." in tensor_name or ".self_attention.v." in tensor_name) and tensor_size == 524288:
            shape = (256, 2048)
        elif (".ff_layer.ffn_layer1_gate." in tensor_name or ".ff_layer.ffn_layer1." in tensor_name) and tensor_size == 25165824:
            shape = (12288, 2048)
        elif ".ff_layer.ffn_layer2." in tensor_name and tensor_size == 25165824:
            shape = (2048, 12288)
        elif "params.lm.softmax.logits_ffn.w" == tensor_name and tensor_size == 524550144:
            shape = (256128, 2048)
        elif "layer_norm" in tensor_name and tensor_size == 2048:
            shape = (1, 1, 2048)

        tensor_dims[tensor_name] = shape

    tensor_dict = {}

    for tensor_name, tensor in fp32_tensors.items():
        quantized_buf_meta = model.Buffers(tensor.Buffer())
        dims = tensor_dims[tensor_name]
        target_name = update_target_name(tensor_name)
        tensor_data = torch.frombuffer(buffer=buf, dtype=torch.float32, offset=quantized_buf_meta.Offset(), count=quantized_buf_meta.Size() // 4)
        if dims is not None:
            tensor_data.reshape(dims)
        tensor_dict[target_name] = tensor_data.to(dtype=TARGET_DTYPE) if TARGET_DTYPE != torch.float32 else tensor_data

    del fp32_tensors

    pool = Pool()
    tasks = [(tensor_name, tensor, buf, scale_tensors, tensor_dims, model, TARGET_DTYPE) for tensor_name, tensor in {**i4_tensors, **i8_tensors}.items()]
    results = pool.map(process_tensor, tasks)
    pool.close()
    pool.join()

    for target_name, tensor_data in results:
        tensor_dict[target_name] = tensor_data

    del i4_tensors, i8_tensors, scale_tensors, buf, model, graph
    input_file.close()

    st.save_file(tensor_dict, sys.argv[2])

if __name__ == "__main__":
    main()
