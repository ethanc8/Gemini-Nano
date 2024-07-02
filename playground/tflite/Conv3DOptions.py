# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
from typing import Any
np = import_numpy()

class Conv3DOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset: int = 0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Conv3DOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsConv3DOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def Conv3DOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # Conv3DOptions
    def Init(self, buf: bytes, pos: int):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Conv3DOptions
    def Padding(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def StrideD(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def StrideW(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def StrideH(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def DilationDFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Conv3DOptions
    def DilationWFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Conv3DOptions
    def DilationHFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

def Conv3DOptionsStart(builder: flatbuffers.Builder):
    builder.StartObject(8)

def Start(builder: flatbuffers.Builder):
    Conv3DOptionsStart(builder)

def Conv3DOptionsAddPadding(builder: flatbuffers.Builder, padding: int):
    builder.PrependInt8Slot(0, padding, 0)

def AddPadding(builder: flatbuffers.Builder, padding: int):
    Conv3DOptionsAddPadding(builder, padding)

def Conv3DOptionsAddStrideD(builder: flatbuffers.Builder, strideD: int):
    builder.PrependInt32Slot(1, strideD, 0)

def AddStrideD(builder: flatbuffers.Builder, strideD: int):
    Conv3DOptionsAddStrideD(builder, strideD)

def Conv3DOptionsAddStrideW(builder: flatbuffers.Builder, strideW: int):
    builder.PrependInt32Slot(2, strideW, 0)

def AddStrideW(builder: flatbuffers.Builder, strideW: int):
    Conv3DOptionsAddStrideW(builder, strideW)

def Conv3DOptionsAddStrideH(builder: flatbuffers.Builder, strideH: int):
    builder.PrependInt32Slot(3, strideH, 0)

def AddStrideH(builder: flatbuffers.Builder, strideH: int):
    Conv3DOptionsAddStrideH(builder, strideH)

def Conv3DOptionsAddFusedActivationFunction(builder: flatbuffers.Builder, fusedActivationFunction: int):
    builder.PrependInt8Slot(4, fusedActivationFunction, 0)

def AddFusedActivationFunction(builder: flatbuffers.Builder, fusedActivationFunction: int):
    Conv3DOptionsAddFusedActivationFunction(builder, fusedActivationFunction)

def Conv3DOptionsAddDilationDFactor(builder: flatbuffers.Builder, dilationDFactor: int):
    builder.PrependInt32Slot(5, dilationDFactor, 1)

def AddDilationDFactor(builder: flatbuffers.Builder, dilationDFactor: int):
    Conv3DOptionsAddDilationDFactor(builder, dilationDFactor)

def Conv3DOptionsAddDilationWFactor(builder: flatbuffers.Builder, dilationWFactor: int):
    builder.PrependInt32Slot(6, dilationWFactor, 1)

def AddDilationWFactor(builder: flatbuffers.Builder, dilationWFactor: int):
    Conv3DOptionsAddDilationWFactor(builder, dilationWFactor)

def Conv3DOptionsAddDilationHFactor(builder: flatbuffers.Builder, dilationHFactor: int):
    builder.PrependInt32Slot(7, dilationHFactor, 1)

def AddDilationHFactor(builder: flatbuffers.Builder, dilationHFactor: int):
    Conv3DOptionsAddDilationHFactor(builder, dilationHFactor)

def Conv3DOptionsEnd(builder: flatbuffers.Builder) -> int:
    return builder.EndObject()

def End(builder: flatbuffers.Builder) -> int:
    return Conv3DOptionsEnd(builder)
