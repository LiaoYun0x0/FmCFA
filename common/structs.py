import torch
import abc
import sys

if sys.version_info >= (3, 7):
    class NpArray:
        def __class_getitem__(cls, args):
            pass
else:
    class _NpArray:
        def __getitem__(self, idx):
            pass

    NpArray = _NpArray()

