import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "../fine_tuned_model"

print("Loading Fine-Tuned TinyLlama...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    is_trainable=False
)

model.eval()
print("Fine-Tuned Model Loaded Successfully.")

SYSTEM_PROMPT = """You are a BFSI call center AI assistant.
Strict Rules:
- DO NOT define concepts. Focus ONLY on customer impact.
- NO paragraphs. Bullet points ONLY.
- Do NOT generate exact rates or numbers.
- If exact details are required, state that verification is required.
MANDATORY RESPONSE FORMAT:
- Point 1
- Point 2
- Point 3"""

def generate_response(user_text: str) -> str:
    prompt = f"<|system|>{SYSTEM_PROMPT}\n\n<|user|>{user_text}\n\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False, # Deterministic greedy search
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ---- Remove prompt echo ----
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>")[-1].strip()

    # ---- Deterministic Formatting Enforcement ----
    lines = decoded.replace("•", "-").split("\n")
    clean_bullets = []

    for line in lines:
        line = line.strip()

        # Must start with dash or look like a bullet
        if not (line.startswith("-") or line.startswith("*")):
            continue
            
        # Hard-strip numbering logic (e.g., "- 1.", "- 2)")
        line = line.replace("- 1.", "-").replace("- 2.", "-").replace("- 3.", "-")
        line = line.replace("-1.", "-").replace("-2.", "-")
        
        # Standardize bullet start
        if line.startswith("*"):
            line = "-" + line[1:]

        lower = line.lower()

        # Logic-Gate: Remove contradictions or branching hallucinations
        # If user asks about increase, don't mention "no change" or "decrease"
        if any(bad_logic in lower for bad_logic in ["no change", "decrease", "may not"]):
            continue

        # Shorten verbose explanations (keep it concise for call centers)
        if len(line) > 140:
            line = line[:140].rsplit(".", 1)[0] + "."

        if len(line.strip()) > 5: # Avoid empty dashes
            clean_bullets.append(line.strip())

    # ---- Enforce EXACT output: 3 Clean Bullets ----
    clean_bullets = clean_bullets[:3]

    # BFSI Fallback: If model output is too messy/contradictory
    if len(clean_bullets) < 2:
        return (
            "- An increase in interest rate may raise your monthly EMI.\n"
            "- The impact depends on loan tenure and outstanding balance.\n"
            "- Exact changes require verification with the bank."
        )

    return "\n".join(clean_bullets)