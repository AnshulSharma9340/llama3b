import os
import torch
import json
import glob
from transformers import AutoTokenizer

def check_setup():
    print("🏥 STARTING SANITY CHECK...\n")

    # 1. CHECK GPU
    print("1️⃣  Checking Hardware...")
    if torch.cuda.is_available():
        print(f"   ✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   ✅ CUDA Version: {torch.version.cuda}")
    else:
        print("   ❌ NO GPU DETECTED! Training will fail.")
        return

    # 2. CHECK IMPORTS
    print("\n2️⃣  Checking Libraries...")
    try:
        import bitsandbytes
        import peft
        import trl
        print("   ✅ All ML libraries (bitsandbytes, peft, trl) are installed.")
    except ImportError as e:
        print(f"   ❌ Missing Library: {e}")
        return

    # 3. CHECK DATA FILES
    print("\n3️⃣  Checking Data Files...")
    files = glob.glob("sampled_data/*.jsonl")
    if not files:
        print("   ❌ No files found in 'sample_data/'!")
    else:
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    first_line = json.loads(file.readline())
                    if "instruction" in first_line and "output" in first_line:
                        print(f"   ✅ {os.path.basename(f)}: Structure OK")
                    else:
                        print(f"   ⚠️ {os.path.basename(f)}: Missing keys! Found: {first_line.keys()}")
            except Exception as e:
                print(f"   ❌ {os.path.basename(f)}: JSON Error ({e})")

    # 4. CHECK TOKENIZER (Hugging Face Login)
    print("\n4️⃣  Checking Tokenizer Access...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        print("   ✅ Tokenizer downloaded successfully (Login is working).")
    except Exception as e:
        print(f"   ❌ Tokenizer Error: {e}")
        print("      (Did you run 'huggingface-cli login'?)")

    print("\n🏁 SANITY CHECK COMPLETE.")

if __name__ == "__main__":
    check_setup()