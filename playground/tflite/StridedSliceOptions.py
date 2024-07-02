# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
from typing import Any
np = import_numpy()

class StridedSliceOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset: int = 0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = StridedSliceOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsStridedSliceOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def StridedSliceOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # StridedSliceOptions
    def Init(self, buf: bytes, pos: int):
        self._tab = flatbuffers.table.Table(buf, pos)

    # StridedSliceOptions
    def BeginMask(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # StridedSliceOptions
    def EndMask(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # StridedSliceOptions
    def EllipsisMask(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # StridedSliceOptions
    def NewAxisMask(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # StridedSliceOptions
    def ShrinkAxisMask(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # StridedSliceOptions
    def Offset(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def StridedSliceOptionsStart(builder: flatbuffers.Builder):
    builder.StartObject(6)

def Start(builder: flatbuffers.Builder):
    StridedSliceOptionsStart(builder)

def StridedSliceOptionsAddBeginMask(builder: flatbuffers.Builder, beginMask: int):
    builder.PrependInt32Slot(0, beginMask, 0)

def AddBeginMask(builder: flatbuffers.Builder, beginMask: int):
    StridedSliceOptionsAddBeginMask(builder, beginMask)

def StridedSliceOptionsAddEndMask(builder: flatbuffers.Builder, endMask: int):
    builder.PrependInt32Slot(1, endMask, 0)

def AddEndMask(builder: flatbuffers.Builder, endMask: int):
    StridedSliceOptionsAddEndMask(builder, endMask)

def StridedSliceOptionsAddEllipsisMask(builder: flatbuffers.Builder, ellipsisMask: int):
    builder.PrependInt32Slot(2, ellipsisMask, 0)

def AddEllipsisMask(builder: flatbuffers.Builder, ellipsisMask: int):
    StridedSliceOptionsAddEllipsisMask(builder, ellipsisMask)

def StridedSliceOptionsAddNewAxisMask(builder: flatbuffers.Builder, newAxisMask: int):
    builder.PrependInt32Slot(3, newAxisMask, 0)

def AddNewAxisMask(builder: flatbuffers.Builder, newAxisMask: int):
    StridedSliceOptionsAddNewAxisMask(builder, newAxisMask)

def StridedSliceOptionsAddShrinkAxisMask(builder: flatbuffers.Builder, shrinkAxisMask: int):
    builder.PrependInt32Slot(4, shrinkAxisMask, 0)

def AddShrinkAxisMask(builder: flatbuffers.Builder, shrinkAxisMask: int):
    StridedSliceOptionsAddShrinkAxisMask(builder, shrinkAxisMask)

def StridedSliceOptionsAddOffset(builder: flatbuffers.Builder, offset: bool):
    builder.PrependBoolSlot(5, offset, 0)

def AddOffset(builder: flatbuffers.Builder, offset: bool):
    StridedSliceOptionsAddOffset(builder, offset)

def StridedSliceOptionsEnd(builder: flatbuffers.Builder) -> int:
    return builder.EndObject()

def End(builder: flatbuffers.Builder) -> int:
    return StridedSliceOptionsEnd(builder)
