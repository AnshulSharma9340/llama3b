import os
import yaml
import glob
from datasets import load_dataset, concatenate_datasets

# --- CONFIGURATION ---
# Instructions map based on filename keywords
TASK_MAP = {
    "icd10": "You are a professional medical coder. Assign ICD-10-CM codes with explanations.",
    "soap": "Summarize the following dialogue into a structured SOAP note.",
    "chat": "You are a helpful medical assistant. Answer the patient query professionally.",
    "mtsamples": "Write a professional clinical report based on the description.",
    "default": "You are an AI medical assistant."
}

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def process_data():
    cfg = load_config()
    raw_path = cfg['raw_data_dir']
    output_path = cfg['processed_data_dir']
    
    print("\n" + "="*50)
    print("🔄 [Task Mixture Pro] Starting Pipeline")
    print("="*50)
    
    files = glob.glob(os.path.join(raw_path, "*.jsonl"))
    if not files:
        print("❌ CRITICAL: No .jsonl files found in data/raw!")
        return

    train_datasets = []
    val_datasets = []
    
    total_stats = {}

    for file in files:
        filename = os.path.basename(file).lower()
        print(f"\n📂 Processing: {filename}")
        
        # 1. Determine Task Instruction
        instruction = TASK_MAP["default"]
        task_name = "General"
        for key in TASK_MAP:
            if key in filename:
                instruction = TASK_MAP[key]
                task_name = key.upper()
                break
        
        # 2. Load Dataset
        try:
            ds = load_dataset("json", data_files=file, split="train")
        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")
            continue

        original_len = len(ds)

        # 3. CLEANING: Remove empty rows
        ds = ds.filter(lambda x: x['input'] and len(str(x['input']).strip()) > 1)
        ds = ds.filter(lambda x: x['output'] and len(str(x['output']).strip()) > 1)
        
        cleaned_len = len(ds)
        if cleaned_len < original_len:
            print(f"   🧹 Removed {original_len - cleaned_len} empty/bad rows.")

        # 4. ENRICHMENT: Add Instruction & Source
        def enrich_data(example):
            if "instruction" not in example or not example["instruction"]:
                example["instruction"] = instruction
            example["source"] = filename # Track where this came from
            return example

        ds = ds.map(enrich_data)

        # 5. STRATIFIED SPLIT (Split THIS file 90/10)
        # This ensures every task is represented in validation
        split = ds.train_test_split(test_size=0.1, seed=42)
        
        train_datasets.append(split['train'])
        val_datasets.append(split['test'])
        
        # Stats
        total_stats[filename] = {
            "Task": task_name,
            "Train": len(split['train']),
            "Val": len(split['test'])
        }
        print(f"   ✅ Ready: {len(split['train'])} Train | {len(split['test'])} Val")

    # 6. MERGE & SHUFFLE
    print("\n🔀 Merging All Tasks...")
    final_train = concatenate_datasets(train_datasets).shuffle(seed=42)
    final_val = concatenate_datasets(val_datasets).shuffle(seed=42)

    # 7. SAVE
    os.makedirs(output_path, exist_ok=True)
    final_train.to_json(os.path.join(output_path, "train.jsonl"))
    final_val.to_json(os.path.join(output_path, "val.jsonl"))

    # 8. FINAL REPORT
    print("\n" + "="*50)
    print(f"📊 FINAL DATASET REPORT (Saved to {output_path})")
    print(f"{'Filename':<30} | {'Task':<10} | {'Train':<8} | {'Val':<8}")
    print("-" * 65)
    for name, stats in total_stats.items():
        print(f"{name:<30} | {stats['Task']:<10} | {stats['Train']:<8} | {stats['Val']:<8}")
    print("-" * 65)
    print(f"{'TOTAL':<30} | {'MIXED':<10} | {len(final_train):<8} | {len(final_val):<8}")
    print("="*50 + "\n")

if __name__ == "__main__":
    process_data()