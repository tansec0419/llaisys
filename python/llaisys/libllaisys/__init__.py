import os
import sys
import ctypes
from pathlib import Path

from .runtime import load_runtime
from .runtime import LlaisysRuntimeAPI
from .llaisys_types import llaisysDeviceType_t, DeviceType
from .llaisys_types import llaisysDataType_t, DataType
from .llaisys_types import llaisysMemcpyKind_t, MemcpyKind
from .llaisys_types import llaisysStream_t
from .tensor import llaisysTensor_t
from .tensor import load_tensor
from .ops import load_ops
from .qwen2 import load_qwen2, LlaisysQwen2Meta, llaisysQwen2Model_t  # 添加这行


def load_shared_library():
    lib_dir = Path(__file__).parent
    if sys.platform == "win32":
        lib_path = lib_dir / "llaisys.dll"
    elif sys.platform == "darwin":
        lib_path = lib_dir / "libllaisys.dylib"
    else:
        lib_path = lib_dir / "libllaisys.so"
    
    if not lib_path.exists():
        # Try to find in system paths
        if sys.platform == "win32":
            lib_name = "llaisys.dll"
        elif sys.platform == "darwin":
            lib_name = "libllaisys.dylib"
        else:
            lib_name = "libllaisys.so"
        return ctypes.CDLL(lib_name)
    
    return ctypes.CDLL(str(lib_path))


LIB_LLAISYS = load_shared_library()
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)
load_qwen2(LIB_LLAISYS)  # 添加这行


__all__ = [
    "LIB_LLAISYS",
    "LlaisysRuntimeAPI",
    "llaisysStream_t",
    "llaisysTensor_t",
    "llaisysDataType_t",
    "DataType",
    "llaisysDeviceType_t",
    "DeviceType",
    "llaisysMemcpyKind_t",
    "MemcpyKind",
    "LlaisysQwen2Meta",      # 添加这行
    "llaisysQwen2Model_t",   # 添加这行
]