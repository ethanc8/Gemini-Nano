# tflite flatbuffer schemas
import tflite.Model, tflite.SubGraph
import sys

input_file = open(sys.argv[0], "rb")
buf = bytearray(input_file.read())

model: tflite.Model.Model = tflite.Model.Model.GetRootAs(buf)

print(f"Version: {model.Version()}")

graph: tflite.SubGraph.SubGraph = model.Subgraphs(0)

print(f"Name: {graph.Name()}")



