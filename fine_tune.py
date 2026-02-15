import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading base model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

dataset = load_from_disk("../data/training_dataset")

# Tokenize dataset
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"]
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="../fine_tuned_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("Starting fine-tuning...")

trainer.train()

trainer.save_model("../fine_tuned_model")

print("Fine-tuning complete.")
