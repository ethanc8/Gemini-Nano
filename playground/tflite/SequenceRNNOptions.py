# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
from typing import Any
np = import_numpy()

class SequenceRNNOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset: int = 0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SequenceRNNOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSequenceRNNOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def SequenceRNNOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # SequenceRNNOptions
    def Init(self, buf: bytes, pos: int):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SequenceRNNOptions
    def TimeMajor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # SequenceRNNOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # SequenceRNNOptions
    def AsymmetricQuantizeInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def SequenceRNNOptionsStart(builder: flatbuffers.Builder):
    builder.StartObject(3)

def Start(builder: flatbuffers.Builder):
    SequenceRNNOptionsStart(builder)

def SequenceRNNOptionsAddTimeMajor(builder: flatbuffers.Builder, timeMajor: bool):
    builder.PrependBoolSlot(0, timeMajor, 0)

def AddTimeMajor(builder: flatbuffers.Builder, timeMajor: bool):
    SequenceRNNOptionsAddTimeMajor(builder, timeMajor)

def SequenceRNNOptionsAddFusedActivationFunction(builder: flatbuffers.Builder, fusedActivationFunction: int):
    builder.PrependInt8Slot(1, fusedActivationFunction, 0)

def AddFusedActivationFunction(builder: flatbuffers.Builder, fusedActivationFunction: int):
    SequenceRNNOptionsAddFusedActivationFunction(builder, fusedActivationFunction)

def SequenceRNNOptionsAddAsymmetricQuantizeInputs(builder: flatbuffers.Builder, asymmetricQuantizeInputs: bool):
    builder.PrependBoolSlot(2, asymmetricQuantizeInputs, 0)

def AddAsymmetricQuantizeInputs(builder: flatbuffers.Builder, asymmetricQuantizeInputs: bool):
    SequenceRNNOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs)

def SequenceRNNOptionsEnd(builder: flatbuffers.Builder) -> int:
    return builder.EndObject()

def End(builder: flatbuffers.Builder) -> int:
    return SequenceRNNOptionsEnd(builder)
