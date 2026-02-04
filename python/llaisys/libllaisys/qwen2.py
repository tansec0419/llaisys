from ctypes import Structure, POINTER, c_size_t, c_float, c_int64, c_char_p, c_void_p
from .llaisys_types import llaisysDataType_t

class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]

llaisysQwen2Model_t = c_void_p

def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [POINTER(LlaisysQwen2Meta)]
    lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t
    
    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None
    
    lib.llaisysQwen2ModelLoadWeight.argtypes = [
        llaisysQwen2Model_t, c_char_p, c_void_p, POINTER(c_size_t), c_size_t, llaisysDataType_t
    ]
    lib.llaisysQwen2ModelLoadWeight.restype = None
    
    lib.llaisysQwen2ModelResetCache.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelResetCache.restype = None
    
    lib.llaisysQwen2ModelInfer.argtypes = [llaisysQwen2Model_t, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelInfer.restype = c_int64