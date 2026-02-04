"""Run all operator tests"""
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

def run_all_tests(device='cpu'):
    """Run all operator tests"""
    print(f"Running all operator tests on {device}...")
    
    # Import and run each test
    from ops import argmax, embedding, swiglu, rms_norm, rope, linear, self_attention
    
    test_modules = [
        ('Argmax', argmax),
        ('Embedding', embedding),
        ('SwiGLU', swiglu),
        ('RMS Norm', rms_norm),
        ('RoPE', rope),
        ('Linear', linear),
        ('Self-Attention', self_attention),
    ]
    
    for name, module in test_modules:
        print(f"\n{'='*60}")
        print(f"Testing {name}...")
        print('='*60)
        
        # Run the test by executing the module
        import subprocess
        result = subprocess.run(
            [sys.executable, module.__file__, '--device', device],
            capture_output=False
        )
        
        if result.returncode != 0:
            print(f"\n❌ {name} test failed!")
            return False
    
    print(f"\n{'='*60}")
    print("✅ All operator tests passed!")
    print('='*60)
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()
    
    success = run_all_tests(args.device)
    sys.exit(0 if success else 1)