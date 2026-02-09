import yaml
import torch
import inspect
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)

# -------------------------------------------------------------------------
# COMPATIBILITY LAYER
# -------------------------------------------------------------------------
try:
    from trl import SFTTrainer, SFTConfig
    HAS_SFT_CONFIG = True
except ImportError:
    from trl import SFTTrainer
    HAS_SFT_CONFIG = False
    SFTConfig = TrainingArguments

class MedicalTrainer:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self):
        print("🧠 Loading Tokenizer & Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg['model_name'], 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # 1. Force 4-bit Compute to Float16 (Strict)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.cfg['use_4bit'],
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, 
            bnb_4bit_use_double_quant=True,
        )

        # 2. Load Model 
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg['model_name'],
            quantization_config=bnb_config,
            device_map="auto",
            use_cache=False,
            torch_dtype=torch.float16 # Request FP16
        )
        
        # 3. Prepare for Training (Important step)
        self.model = prepare_model_for_kbit_training(self.model)

        # 4. NUCLEAR SANITIZATION ☢️
        # We iterate over every module and FORCE compatible types.
        print("💉 FORCE-CONVERTING all modules to RTX 2050 compatible types...")
        for name, module in self.model.named_modules():
            # Force Normalization layers to Float32 (Safer & Fixes BF16 crash)
            if "norm" in name.lower():
                module.to(torch.float32)

        # Force any remaining BFloat16 params to Float16
        for name, param in self.model.named_parameters():
            if param.dtype == torch.bfloat16:
                print(f"   Converting {name} from BF16 -> FP16")
                param.data = param.data.to(torch.float16)

        # Force config to report FP16
        self.model.config.torch_dtype = torch.float16
        self.model.gradient_checkpointing_enable()

    def get_lora_config(self):
        print("⚙️ Creating LoRA config...")
        return LoraConfig(
            r=self.cfg['lora_r'],
            lora_alpha=self.cfg['lora_alpha'],
            lora_dropout=self.cfg['lora_dropout'],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

    # --- MANUAL DATA FORMATTING ---
    def apply_chat_template(self, batch):
        texts = []
        for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{inst}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{inp}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{out}<|eot_id|>"
            texts.append(prompt)
        return {"text": texts}

    def train(self):
        self.load_model_and_tokenizer()
        peft_config = self.get_lora_config()

        print(f"📚 Loading dataset from {self.cfg['processed_data_dir']}")
        train_dataset = load_dataset("json", data_files=f"{self.cfg['processed_data_dir']}/train.jsonl", split="train")
        val_dataset = load_dataset("json", data_files=f"{self.cfg['processed_data_dir']}/val.jsonl", split="train")

        print("🛠️  Pre-formatting dataset manually...")
        train_dataset = train_dataset.map(self.apply_chat_template, batched=True)
        val_dataset = val_dataset.map(self.apply_chat_template, batched=True)

        # --- ARGUMENTS (FIXED FOR QLORA + FP16) ---
        args_dict = {
            "output_dir": self.cfg['output_dir'],
            "num_train_epochs": self.cfg['epochs'],
            "per_device_train_batch_size": self.cfg['batch_size'],
            "gradient_accumulation_steps": self.cfg['grad_accumulation'],
            "learning_rate": self.cfg['learning_rate'],
            "optim": "paged_adamw_8bit",  
            "logging_steps": 10,
            "save_steps": 50,
            "report_to": "none",
            "eval_strategy": "steps", 
            "eval_steps": 50,
            "fp16": False,  # ✅ CRITICAL FIX: Disable native FP16 to avoid gradient scaler
            "bf16": False,  
            "group_by_length": True,
            "gradient_checkpointing": True,
            "max_grad_norm": 0.3,  # ✅ Added: helps with gradient stability
            "warmup_ratio": 0.03,  # ✅ Added: gradual warmup
        }

        # --- CONFIG OBJECT CREATION ---
        if HAS_SFT_CONFIG:
            try:
                args = SFTConfig(**args_dict, max_seq_length=self.cfg['max_seq_length'], packing=False, dataset_text_field="text")
            except TypeError as e:
                print(f"⚠️ SFTConfig mismatch ({e}) — forcing fields")
                if "evaluation_strategy" in str(e):
                    args_dict["evaluation_strategy"] = args_dict.pop("eval_strategy")
                
                args = SFTConfig(**args_dict)
                args.max_seq_length = self.cfg['max_seq_length']
                args.packing = False
                args.dataset_text_field = "text"
        else:
            args_dict["evaluation_strategy"] = args_dict.pop("eval_strategy")
            args = TrainingArguments(**args_dict)

        # --- TRAINER SETUP ---
        trainer_kwargs = {
            "model": self.model,
            "train_dataset": train_dataset,
            "eval_dataset": val_dataset,
            "peft_config": peft_config,
            "args": args,
            "formatting_func": None, 
        }

        sig = inspect.signature(SFTTrainer.__init__)
        if "processing_class" in sig.parameters:
            trainer_kwargs["processing_class"] = self.tokenizer
        else:
            trainer_kwargs["tokenizer"] = self.tokenizer

        if "max_seq_length" in sig.parameters:
             trainer_kwargs["max_seq_length"] = self.cfg['max_seq_length']
             trainer_kwargs["packing"] = False
             trainer_kwargs["dataset_text_field"] = "text"

        print("\n🚀 STARTING TRAINING NOW...")
        trainer = SFTTrainer(**trainer_kwargs)
        
        # Double check model is sanitized before loop starts
        for name, param in trainer.model.named_parameters():
             if param.dtype == torch.bfloat16:
                 param.data = param.data.to(torch.float16)
        
        trainer.train()
        
        print(f"💾 Saving Model to {self.cfg['output_dir']}/final_adapter")
        trainer.model.save_pretrained(f"{self.cfg['output_dir']}/final_adapter")
        self.tokenizer.save_pretrained(f"{self.cfg['output_dir']}/final_adapter")

if __name__ == "__main__":
    trainer = MedicalTrainer()
    trainer.train()