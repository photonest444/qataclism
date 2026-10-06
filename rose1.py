
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

# --- КОНФИГУРАЦИЯ QATACLISM ROSE 7B ---
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # 4-бит для T4
OUTPUT_DIR = "./Qataclism-Rose-7B-Stage1"
MAX_SEQ_LENGTH = 2048  # Qwen2.5-7B нормально тянет 2048 на T4 [citation:3]

# --- ЗАГРУЗКА МОДЕЛИ (QLoRA) ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,  # QLoRA для экономии VRAM
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Стандартный ранг для QLoRA [citation:3][citation:15]
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
            f"<|im_start|>system\nYou are Qataclism 1.0 Rose 7B, an expert backend developer.<|im_end|>\n"
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
        per_device_train_batch_size=1,  # T4-safe для 7B [citation:9]
        gradient_accumulation_steps=8,  # Эффективный батч = 8
        warmup_steps=5,
        num_train_epochs=3,  # 3 эпохи на 202 примера — разумно [citation:6]
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

print("Запуск обучения Qataclism 1.0 Rose 7B (этап 1)...")
trainer.train()

# --- СОХРАНЕНИЕ LoRA-АДАПТЕРА ---
print("Сохранение адаптера...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Готово! Адаптер сохранён в {OUTPUT_DIR}")