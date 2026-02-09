import torch
from transformers import AutoTokenizer

def get_tokenizer(model_name):
    print(f"🔤 Loading Tokenizer for {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        padding_side="right" # Standard for Training (SFTTrainer)
    )

    # --- LLAMA 3 SPECIFIC FIXES ---
    # Llama 3 does not have a default pad token. We map it to the EOS token.
    # This prevents the "ValueError: Asking to pad but the tokenizer does not have a padding token" error.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("   ⚠️ Pad Token was None. Mapped to EOS Token.")

    # Ensure model knows the max length to avoid OOM on small GPUs
    if not hasattr(tokenizer, "model_max_length") or tokenizer.model_max_length > 100000:
        tokenizer.model_max_length = 512
        print("   ⚠️ Enforced max_length = 512 for memory safety.")

    return tokenizer

# --- TEST FUNCTION ---
if __name__ == "__main__":
    # You can run this file directly to test the tokenizer!
    t = get_tokenizer("meta-llama/Llama-3.2-3B-Instruct")
    sample_text = "Hello doctor, I have pain."
    tokens = t(sample_text)
    print("\n✅ Tokenizer Test Success!")
    print(f"Sample: '{sample_text}'")
    print(f"IDs: {tokens['input_ids']}")