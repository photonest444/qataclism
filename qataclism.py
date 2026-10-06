
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

# --- КОНФИГУРАЦИЯ ---
MODEL_NAME = "Qwen/Qwen3.5-0.8B"  # Самая маленькая официальная модель
OUTPUT_DIR = "./Qataclism-Rose-Stage1"
MAX_SEQ_LENGTH = 2048

# --- ЗАГРУЗКА (bf16 LoRA, не QLoRA!) ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,      # НЕ QLoRA для Qwen3.5
    load_in_16bit=True,      # bf16 LoRA
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# --- ДАТАСЕТ: Techta/backend-code-generator-dataset ---
dataset = load_dataset("Techta/backend-code-generator-dataset", split="train")
print(f"Исходный размер: {len(dataset)}")

def formatting_prompts_func(examples):
    texts = []
    for desc, reqs, files, framework, language in zip(
        examples["description"], examples["requirements"],
        examples["code_files"], examples["framework"], examples["language"]
    ):
        task = (
            f"Create a {framework} backend project.\n"
            f"Language: {language}\n"
            f"Description: {desc}\n"
            f"Requirements:\n" + "\n".join(f"- {r}" for r in reqs)
        )
        code_parts = []
        if files:
            for fn, content in files.items():
                if content:
                    ext = fn.split(".")[-1].lower()
                    lang_map = {"py": "python", "js": "javascript", "ts": "typescript",
                                "json": "json", "yml": "yaml", "yaml": "yaml",
                                "md": "markdown", "txt": "text", "env": "text"}
                    code_parts.append(f"### {fn}\n```{lang_map.get(ext, 'text')}\n{content}\n```")
        if not code_parts:
            texts.append("")
            continue
        output = "\n\n".join(code_parts)
        text = (
            f"<|im_start|>system\nYou are Qataclism Rose, an expert backend developer.<|im_end|>\n"
            f"<|im_start|>user\n{task}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
dataset = dataset.filter(lambda x: len(x["text"]) > 100)
print(f"После фильтрации: {len(dataset)}")

# --- ТРЕНЕР ---
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

print("Запуск обучения Qataclism Rose (этап 1)...")
trainer.train()

# --- СОХРАНЕНИЕ LoRA-АДАПТЕРА ---
print("Сохранение адаптера...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Готово! Адаптер сохранён в {OUTPUT_DIR}")