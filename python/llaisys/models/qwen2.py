from typing import Sequence, List
from ..libllaisys import LIB_LLAISYS, LlaisysQwen2Meta, DataType, DeviceType
from ctypes import c_size_t, c_int64, c_char_p, c_void_p, pointer
import numpy as np
import json
import safetensors
from pathlib import Path
from tqdm import tqdm
import torch


class Qwen2:
    """Qwen2 model for text generation using LLAISYS backend"""

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        # Load config
        config_path = model_path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Use F32 for computation (convert all weights)
        model_dtype = DataType.F32
        
        print(f"Using computation dtype: F32")
        
        # Create model meta
        meta = LlaisysQwen2Meta()
        meta.dtype = model_dtype
        meta.nlayer = config.get("num_hidden_layers", 28)
        meta.hs = config.get("hidden_size", 1536)
        meta.nh = config.get("num_attention_heads", 12)
        meta.nkvh = config.get("num_key_value_heads", 2)
        meta.dh = meta.hs // meta.nh
        meta.di = config.get("intermediate_size", 8960)
        meta.maxseq = config.get("max_position_embeddings", 131072)
        meta.voc = config.get("vocab_size", 151936)
        meta.epsilon = config.get("rms_norm_eps", 1e-6)
        meta.theta = config.get("rope_theta", 10000.0)
        meta.end_token = config.get("eos_token_id", 151643)
        
        self._meta = meta
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(pointer(meta))
        self._eos_token_id = meta.end_token
        
        print(f"Loading model from local path: {model_path}")
        
        # Load weights
        safetensor_files = sorted(model_path.glob("*.safetensors"))
        
        # Collect all weight names first for progress bar
        all_weights = []
        for file in safetensor_files:
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            all_weights.extend(data_.keys())
        
        with tqdm(total=len(all_weights), desc="Loading weights") as pbar:
            for file in safetensor_files:
                data_ = safetensors.safe_open(file, framework="pt", device="cpu")
                for name_ in data_.keys():
                    # Load as torch tensor and convert to float32
                    tensor = data_.get_tensor(name_).float()
                    tensor_np = tensor.numpy()
                    
                    # Ensure contiguous
                    tensor_np = np.ascontiguousarray(tensor_np)
                    shape = tensor_np.shape
                    
                    # Create shape array
                    shape_arr = (c_size_t * len(shape))(*shape)
                    
                    # Load weight (always use F32)
                    LIB_LLAISYS.llaisysQwen2ModelLoadWeight(
                        self._model,
                        name_.encode('utf-8'),
                        tensor_np.ctypes.data_as(c_void_p),
                        shape_arr,
                        c_size_t(len(shape)),
                        DataType.F32
                    )
                    
                    pbar.set_postfix_str(f"Loaded {name_}")
                    pbar.update(1)

    def __del__(self):
        if hasattr(self, '_model') and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> List[int]:
        """
        Generate tokens using greedy decoding (argmax)
        """
        # Reset KV cache for new generation
        LIB_LLAISYS.llaisysQwen2ModelResetCache(self._model)
        
        # Convert inputs to list if needed
        output_tokens = list(inputs)
        
        # Create input array
        input_arr = (c_int64 * len(inputs))(*inputs)
        
        # Prefill phase
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model,
            input_arr,
            c_size_t(len(inputs))
        )
        output_tokens.append(next_token)
        
        # Decode phase
        for _ in range(max_new_tokens - 1):
            if next_token == self._eos_token_id:
                break
            
            single_token = (c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                single_token,
                c_size_t(1)
            )
            output_tokens.append(next_token)
        
        return output_tokens