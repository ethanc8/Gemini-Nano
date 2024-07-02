# tflite flatbuffer schemas
import tflite.Model, tflite.SubGraph
import sys

from mediapipe.tasks.cc.genai.inference.proto import llm_params_pb2

input_file = open(sys.argv[1], "rb")
buf = bytearray(input_file.read())

model: tflite.Model.Model = tflite.Model.Model.GetRootAs(buf)

print(f"tflite file version: {model.Version()}")

graph: tflite.SubGraph.SubGraph = model.Subgraphs(0)

print(f"Model name: {graph.Name()}")

print("===METADATA===")

for i in range(model.MetadataLength()):
    metametadata = model.Metadata(i)
    metadata_buf = model.Buffers(metametadata.Buffer())
    print(metametadata.Name().decode("utf-8"))
    # Get the buffer data
    if metadata_buf.Size() < 80: # Don't print huge buffers
        print(buf[metadata_buf.Offset():metadata_buf.Offset()+metadata_buf.Size()])

sys.exit(0)

print("===TENSORS===")

nameOfTensorType = {
     0: "FLOAT32",
     9: "INT8   ",
    17: "INT4   ",
}

for i in range(graph.TensorsLength()):
    tensor = graph.Tensors(i)
    print(f"{nameOfTensorType[tensor.Type()]} {tensor.Shape(0)} {tensor.Name().decode("utf-8")}")

