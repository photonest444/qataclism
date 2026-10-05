
import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# --- КОНФИГУРАЦИЯ QATACLISM 1.0 ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"  # Открытая модель, скачается без токена
DATASET_NAME = "Techta/backend-code-generator-dataset"
OUTPUT_DIR = "./Qataclism-1.0"
MAX_SEQ_LENGTH = 1024

# --- ЗАГРУЗКА МОДЕЛИ (QLoRA) ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# --- НАСТРОЙКА LoRA ---
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

# --- ЗАГРУЗКА ДАТАСЕТА ---
dataset = load_dataset(DATASET_NAME, split="train")

# --- ФОРМАТИРОВАНИЕ ---
print("Колонки датасета:", dataset.column_names)
print("Пример данных:", dataset[0])
def formatting_prompts_func(examples):
    texts = []
    for desc, reqs, files, framework, language in zip(
        examples["description"],
        examples["requirements"],
        examples["code_files"],
        examples["framework"],
        examples["language"],
    ):
        # Собираем только Python-файлы (если датасет смешанный)
        if language != "python":
            continue
        
        # Формируем "задачу" из описания и требований
        task = f"Create a {framework} backend project.\n\nDescription: {desc}\n\nRequirements:\n" + "\n".join(f"- {r}" for r in reqs)
        
        # Формируем "ответ" из файлов кода
        code_parts = []
        for filename, content in files.items():
            if content and filename.endswith(".py"):
                code_parts.append(f"### {filename}\n```python\n{content}\n```")
        
        if not code_parts:
            continue
        
        output = "\n\n".join(code_parts)
        
        text = (
            f"<|im_start|>system\nYou are Qataclism 1.0, an expert backend developer specializing in Flask and FastAPI.<|im_end|>\n"
            f"<|im_start|>user\n{task}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        texts.append(text)
    
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# --- ТРЕНЕР ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
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

# --- ЗАПУСК ---
print("Запуск обучения Qataclism 1.0...")
trainer.train()

# --- СОХРАНЕНИЕ ---
print("Сохранение Qataclism 1.0...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Модель сохранена в {OUTPUT_DIR}")