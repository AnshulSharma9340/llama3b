import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

class MedicalInferencer:
    def __init__(self):
        self.cfg = load_config()
        self.model = None
        self.tokenizer = None
        self.setup()

    def setup(self):
        print("🧠 Loading Base Model & Adapters...")
        
        # Load Base Model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.cfg['model_name'],
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg['model_name'])
        
        # Load Trained Adapters
        adapter_path = f"{self.cfg['output_dir']}/final_adapter"
        print(f"🔗 Attaching Adapters from: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)

    def chat(self, user_query):
        # Official Llama 3 Prompt Format for Inference
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a knowledgeable medical assistant. Provide accurate, helpful information.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's reply
        clean_response = response.split("assistant")[-1].strip()
        return clean_response

if __name__ == "__main__":
    bot = MedicalInferencer()
    print("\n✅ Medical AI Ready! Type 'exit' to quit.\n")
    
    while True:
        q = input("User: ")
        if q.lower() == "exit": break
        print(f"AI: {bot.chat(q)}\n")