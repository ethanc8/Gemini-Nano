# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
from typing import Any
np = import_numpy()

class HashtableOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset: int = 0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = HashtableOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsHashtableOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def HashtableOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # HashtableOptions
    def Init(self, buf: bytes, pos: int):
        self._tab = flatbuffers.table.Table(buf, pos)

    # HashtableOptions
    def TableId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # HashtableOptions
    def KeyDtype(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # HashtableOptions
    def ValueDtype(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def HashtableOptionsStart(builder: flatbuffers.Builder):
    builder.StartObject(3)

def Start(builder: flatbuffers.Builder):
    HashtableOptionsStart(builder)

def HashtableOptionsAddTableId(builder: flatbuffers.Builder, tableId: int):
    builder.PrependInt32Slot(0, tableId, 0)

def AddTableId(builder: flatbuffers.Builder, tableId: int):
    HashtableOptionsAddTableId(builder, tableId)

def HashtableOptionsAddKeyDtype(builder: flatbuffers.Builder, keyDtype: int):
    builder.PrependInt8Slot(1, keyDtype, 0)

def AddKeyDtype(builder: flatbuffers.Builder, keyDtype: int):
    HashtableOptionsAddKeyDtype(builder, keyDtype)

def HashtableOptionsAddValueDtype(builder: flatbuffers.Builder, valueDtype: int):
    builder.PrependInt8Slot(2, valueDtype, 0)

def AddValueDtype(builder: flatbuffers.Builder, valueDtype: int):
    HashtableOptionsAddValueDtype(builder, valueDtype)

def HashtableOptionsEnd(builder: flatbuffers.Builder) -> int:
    return builder.EndObject()

def End(builder: flatbuffers.Builder) -> int:
    return HashtableOptionsEnd(builder)
