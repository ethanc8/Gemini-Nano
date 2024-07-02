# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
from typing import Any
np = import_numpy()

class EmbeddingLookupSparseOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset: int = 0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = EmbeddingLookupSparseOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsEmbeddingLookupSparseOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def EmbeddingLookupSparseOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # EmbeddingLookupSparseOptions
    def Init(self, buf: bytes, pos: int):
        self._tab = flatbuffers.table.Table(buf, pos)

    # EmbeddingLookupSparseOptions
    def Combiner(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def EmbeddingLookupSparseOptionsStart(builder: flatbuffers.Builder):
    builder.StartObject(1)

def Start(builder: flatbuffers.Builder):
    EmbeddingLookupSparseOptionsStart(builder)

def EmbeddingLookupSparseOptionsAddCombiner(builder: flatbuffers.Builder, combiner: int):
    builder.PrependInt8Slot(0, combiner, 0)

def AddCombiner(builder: flatbuffers.Builder, combiner: int):
    EmbeddingLookupSparseOptionsAddCombiner(builder, combiner)

def EmbeddingLookupSparseOptionsEnd(builder: flatbuffers.Builder) -> int:
    return builder.EndObject()

def End(builder: flatbuffers.Builder) -> int:
    return EmbeddingLookupSparseOptionsEnd(builder)
