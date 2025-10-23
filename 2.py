import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"  # أو يمكنك تحديد device manually
    ).to(device)
    model.eval()
    return tokenizer, model

def generate_with_flash_attention(tokenizer, model, prompt, max_new_tokens=256, temperature=0.6):
    device = next(model.parameters()).device
    print("Device for model:", device)
    # تأكيد حالة Flash SDP
    print("flash_sdp_enabled (before):", torch.backends.cuda.flash_sdp_enabled())
    print("mem_efficient_sdp_enabled (before):", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math_sdp_enabled (before):", torch.backends.cuda.math_sdp_enabled())

    # ترميز النص
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    # تفعيل Flash Attention إن أمكن
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        print("Inside context — flash_sdp_enabled:", torch.backends.cuda.flash_sdp_enabled())
        # استدعاء النموذج للتوليد
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )

    # فك الشفرة وعرض النص الناتج
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/m/huggingface/Qwen3-0.6B"
    tokenizer, model = load_model(model_path, device)
    prompt = "Hello, how are you today?"
    result = generate_with_flash_attention(tokenizer, model, prompt, max_new_tokens=128, temperature=0.7)
    print("Generated output:\n", result)

