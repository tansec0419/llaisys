import argparse
import sys
import time
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from llaisys.models import Qwen2


def load_hf_model(model_path, device):
    """加载 HuggingFace 参考模型（可选）"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print(f"Loading HuggingFace model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        device_map = "cpu" if device == "cpu" else "cuda"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
        )
        
        return tokenizer, model, model_path
    except Exception as e:
        print(f"Warning: Could not load HuggingFace model: {e}")
        print("Will only test LLAISYS implementation")
        return None, None, model_path


def test_llaisys_model(model_path, prompt="Who are you?", max_tokens=128):
    """测试 LLAISYS 实现的模型"""
    from transformers import AutoTokenizer
    
    print("\n" + "="*60)
    print("Testing LLAISYS Implementation")
    print("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Tokenize input
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt")
    input_ids = inputs.input_ids[0].tolist()
    
    print(f"\nPrompt: {prompt}")
    print(f"Input tokens: {input_ids[:10]}... (length: {len(input_ids)})")
    
    # Load LLAISYS model
    print("\nLoading LLAISYS model...")
    model = Qwen2(model_path)
    
    # Generate
    print("Generating...")
    start_time = time.time()
    output_ids = model.generate(input_ids, max_new_tokens=max_tokens)
    elapsed = time.time() - start_time
    
    # Decode
    output_text = tokenizer.decode(output_ids, skip_special_tokens=False)
    
    print("\n=== LLAISYS Result ===")
    print(f"\nTokens ({len(output_ids)}):")
    print(output_ids[:50])  # 只打印前50个
    if len(output_ids) > 50:
        print("...")
    
    print(f"\nGenerated text:")
    print(output_text)
    print(f"\nTime elapsed: {elapsed:.2f}s")
    print(f"Tokens/sec: {len(output_ids)/elapsed:.2f}")
    
    return output_ids, output_text


def test_hf_model(tokenizer, model, prompt="Who are you?", max_tokens=128):
    """测试 HuggingFace 参考模型"""
    import torch
    
    print("\n" + "="*60)
    print("Testing HuggingFace Reference")
    print("="*60)
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print("Generating...")
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # greedy decoding
        )
    elapsed = time.time() - start_time
    
    # Get full output (including prompt)
    output_ids_list = generated_ids[0].cpu().tolist()
    
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    
    print("\n=== HuggingFace Result ===")
    print(f"\nTokens ({len(output_ids_list)}):")
    print(output_ids_list[:50])
    if len(output_ids_list) > 50:
        print("...")
    
    print(f"\nGenerated text:")
    print(output_text)
    print(f"\nTime elapsed: {elapsed:.2f}s")
    print(f"Tokens/sec: {len(output_ids_list)/elapsed:.2f}")
    
    return output_ids_list, output_text


def get_default_model_path():
    """获取默认模型路径（用于 CI 环境）"""
    # 常见的模型缓存位置
    possible_paths = [
        # HuggingFace cache (Linux/Mac)
        Path.home() / ".cache" / "huggingface" / "hub",
        # HuggingFace cache (Windows)
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    
    for cache_dir in possible_paths:
        if cache_dir.exists():
            # 查找 DeepSeek-R1-Distill-Qwen-1.5B 模型
            model_dirs = list(cache_dir.glob("models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/*"))
            if model_dirs:
                # 返回第一个找到的 snapshot
                return str(model_dirs[0])
    
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--prompt", type=str, default="Who are you?", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace model test")
    parser.add_argument("--test", action="store_true", help="Run test mode (auto-detect model path)")
    args = parser.parse_args()
    
    # Get model path from multiple sources (优先级从高到低)
    model_path = (
        args.model or                           # 1. 命令行参数
        os.environ.get('MODEL_PATH') or         # 2. 环境变量
        (get_default_model_path() if args.test else None)  # 3. 自动检测（仅在 --test 模式）
    )
    
    if not model_path:
        print("Error: Model path must be provided via one of:")
        print("  1. --model argument")
        print("  2. MODEL_PATH environment variable")
        print("  3. --test flag (auto-detect from HuggingFace cache)")
        print("\nNo model found in any location.")
        sys.exit(1)
    
    print(f"Using model path: {model_path}")
    
    # Verify model path exists
    if not Path(model_path).exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)
    
    # Test LLAISYS implementation
    try:
        llaisys_ids, llaisys_text = test_llaisys_model(
            model_path, 
            prompt=args.prompt, 
            max_tokens=args.max_tokens
        )
    except Exception as e:
        print(f"Error testing LLAISYS model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test HuggingFace reference (optional)
    if not args.skip_hf:
        tokenizer, hf_model, _ = load_hf_model(model_path, args.device)
        if tokenizer and hf_model:
            try:
                hf_ids, hf_text = test_hf_model(
                    tokenizer, 
                    hf_model, 
                    prompt=args.prompt, 
                    max_tokens=args.max_tokens
                )
                
                # Compare results
                print("\n" + "="*60)
                print("Comparison")
                print("="*60)
                
                # Compare tokens
                min_len = min(len(llaisys_ids), len(hf_ids))
                match_count = sum(1 for i in range(min_len) if llaisys_ids[i] == hf_ids[i])
                
                print(f"\nTotal tokens - LLAISYS: {len(llaisys_ids)}, HF: {len(hf_ids)}")
                print(f"Matching tokens: {match_count}/{min_len}")
                print(f"Match rate: {match_count/min_len*100:.1f}%")
                
                # Show first difference
                for i in range(min_len):
                    if llaisys_ids[i] != hf_ids[i]:
                        print(f"\n⚠️ First difference at position {i}:")
                        print(f"  LLAISYS: {llaisys_ids[max(0,i-2):i+3]}")
                        print(f"  HF:      {hf_ids[max(0,i-2):i+3]}")
                        sys.exit(1)  # Fail test if mismatch
                else:
                    print("\n✅ All tokens match!")
            except Exception as e:
                print(f"Warning: Could not test HuggingFace model: {e}")
                print("Continuing with LLAISYS-only test...")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()