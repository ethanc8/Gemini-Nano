# tflite flatbuffer schemas
import tflite.Model, tflite.SubGraph
import sys

from mediapipe.tasks.cc.genai.inference.proto import llm_params_pb2

# FIXME: sentencepiece and mediapipe protobuf conflict due to both setting some abseil flag
# import sentencepiece as spm

input_file = open(sys.argv[1], "rb")
buf = bytearray(input_file.read())

model: tflite.Model.Model = tflite.Model.Model.GetRootAs(buf)

print(f"tflite file version: {model.Version()}")

graph: tflite.SubGraph.SubGraph = model.Subgraphs(0)

print(f"Model name: {graph.Name()}")

print("===METADATA===")

parameters: llm_params_pb2.LlmParameters = None


for i in range(model.MetadataLength()):
    metametadata = model.Metadata(i)
    metadata_buf = model.Buffers(metametadata.Buffer())
    metadata = buf[metadata_buf.Offset():metadata_buf.Offset()+metadata_buf.Size()]
    print(metametadata.Name().decode("utf-8"))
    if metadata_buf.Size() < 80: # Don't print huge buffers
        print(metadata)
    if metametadata.Name() == b'odml.infra.proto.LlmParameters':
        parameters = llm_params_pb2.LlmParameters()
        parameters.ParseFromString(metadata)
        print(parameters)
    elif metametadata.Name() == b'spm_vocab_model':
        # We can load it with sentencepiece like so:
        # sp = spm.SentencePieceProcessor()
        # sp.load_from_serialized_proto(metadata)
        pass
    elif metametadata.Name() == b'odml.infra.LlmModelType':
        print(metadata[0])

# sys.exit(0)

print("===TENSORS===")

nameOfTensorType = {
     0: "FLOAT32",
     9: "INT8   ",
    17: "INT4   ",
}

for i in range(graph.TensorsLength()):
    tensor = graph.Tensors(i)
    print(f"{nameOfTensorType[tensor.Type()]} {tensor.Shape(0)} {tensor.Name().decode("utf-8")}")
    tensor_buf = model.Buffers(tensor.Buffer())
    tensor_data = buf[tensor_buf.Offset():tensor_buf.Offset()+tensor_buf.Size()]
    # print(tensor_data)