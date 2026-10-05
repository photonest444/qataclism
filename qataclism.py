import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# --- КОНФИГУРАЦИЯ QATACLISM 1.0 ---
MODEL_NAME = "huihui-ai/Qwen2.5-Code-0.5B-Instruct-abliterated" # Или "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DATASET_NAME = "Techta/backend-code-generator-dataset"
OUTPUT_DIR = "./Qataclism-1.0"
MAX_SEQ_LENGTH = 1024 # Увеличим, чтобы модель видела больше контекста кода

# --- ЗАГРУЗКА МОДЕЛИ (QLoRA) ---
# load_in_4bit=True критически важен для экономии памяти
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None, # Авто-определение (bf16/fp16)
    load_in_4bit=True, # 4-битное квантование
)

# --- НАСТРОЙКА LoRA ---
model = FastLanguageModel.get_peft_model(
    model,
    r=16, # Ранг LoRA (чем выше, тем больше параметров учится)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth", # Экономия памяти
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# --- ЗАГРУЗКА И ФОРМАТИРОВАНИЕ ДАТАСЕТА ---
dataset = load_dataset(DATASET_NAME, split="train")

# Функция форматирования: превращаем задачу в диалог
def formatting_prompts_func(examples):
    instructions = examples["task"] # Или examples["instruction"], зависит от колонок
    outputs = examples["solution"] # Или examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # Стандартный формат чата Qwen
        text = f"<|im_start|>system\nYou are Qataclism 1.0, an expert backend developer specializing in Flask and FastAPI.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched=True)

# --- ТРЕНЕР ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False, # Для кода лучше False, чтобы не склеивать разные примеры
    args=TrainingArguments(
        per_device_train_batch_size=2, # Маленький батч для экономии памяти
        gradient_accumulation_steps=4, # Эффективный батч = 8
        warmup_steps=5,
        max_steps=60, # Для теста. Для полного обучения увеличьте до 300-500
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit", # 8-битный оптимизатор
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
print(f"Модель успешно обучена и сохранена в {OUTPUT_DIR}")