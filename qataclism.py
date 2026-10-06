
import json
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

# --- КОНФИГУРАЦИЯ QATACLISM LOTUS 1.0 (ЭТАП 1) ---
MODEL_NAME = "Qwen/Qwen3.5-4B"
OUTPUT_DIR = "./Qataclism-Lotus-1.0-Stage1"
MAX_SEQ_LENGTH = 2048
SYSTEM_PROMPT = "You are Qataclism Lotus 1.0, a versatile AI assistant."

# --- ЗАГРУЗКА МОДЕЛИ (bf16 LoRA) ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# --- ЗАГРУЗКА ЛОКАЛЬНОГО ДАТАСЕТА ---
print("Загрузка dataset.jsonl...")
with open("dataset.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

texts = []
for line in lines:
    obj = json.loads(line)
    inst = obj.get("instruction", "")
    inp = obj.get("input", "")
    out = obj.get("output", "")

    if not inst or not out:
        continue

    task = f"{inst}\n{inp}".strip() if inp else inst
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{task}<|im_end|>\n"
        f"<|im_start|>assistant\n{out}<|im_end|>"
    )
    texts.append(text)

dataset = Dataset.from_list([{"text": t} for t in texts])
dataset = dataset.filter(lambda x: len(x["text"]) > 50)
print(f"После фильтрации: {len(dataset)} примеров")

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
        warmup_steps=10,
        num_train_epochs=1,  # <-- ОДНА ЭПОХА
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        save_steps=200,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

print("Запуск обучения Qataclism Lotus 1.0 (этап 1, 1 эпоха)...")
trainer.train()

# --- СОХРАНЕНИЕ LoRA-АДАПТЕРА ---
print("Сохранение адаптера...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Готово! Адаптер сохранён в {OUTPUT_DIR}")