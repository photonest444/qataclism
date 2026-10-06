
import json
import os
from datasets import load_dataset, concatenate_datasets, Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

# --- КОНФИГУРАЦИЯ QATACLISM LOTUS 1.0 ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_DIR = "./Qataclism-Lotus-1.0"
GGUF_DIR = "Qataclism-Lotus-1.0-GGUF"
MAX_SEQ_LENGTH = 2048
SYSTEM_PROMPT = "You are Qataclism Lotus 1.0, a versatile AI assistant skilled in backend development, frontend design, and multilingual conversation."

# --- ЗАГРУЗКА МОДЕЛИ (QLoRA) ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
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

# --- ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ---
def make_chat(user_text, assistant_text):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
    )

# --- 1. Techta/backend-code-generator-dataset ---
print("Загрузка Techta/backend-code-generator-dataset...")
ds_techta = load_dataset("Techta/backend-code-generator-dataset", split="train")
def fmt_techta(examples):
    texts = []
    for desc, reqs, files, framework, language in zip(
        examples["description"], examples["requirements"],
        examples["code_files"], examples["framework"], examples["language"]
    ):
        task = f"Create a {framework} backend project.\nLanguage: {language}\nDescription: {desc}\nRequirements:\n" + "\n".join(f"- {r}" for r in reqs)
        code_parts = []
        if files:
            for fn, content in files.items():
                if content:
                    ext = fn.split(".")[-1].lower()
                    lang_map = {"py": "python", "js": "javascript", "ts": "typescript", "json": "json", "yml": "yaml", "md": "markdown", "txt": "text"}
                    code_parts.append(f"### {fn}\n```{lang_map.get(ext, 'text')}\n{content}\n```")
        if not code_parts:
            texts.append("")
            continue
        texts.append(make_chat(task, "\n\n".join(code_parts)))
    return {"text": texts}
ds_techta = ds_techta.map(fmt_techta, batched=True)
ds_techta = ds_techta.filter(lambda x: len(x["text"]) > 100)
ds_techta = ds_techta.remove_columns([c for c in ds_techta.column_names if c != "text"])
print(f"  → {len(ds_techta)} примеров")

# --- 2. Reubencf/frontend-html-tailwind-js ---
print("Загрузка Reubencf/frontend-html-tailwind-js...")
ds_front = load_dataset("Reubencf/frontend-html-tailwind-js", split="train")
def fmt_front(examples):
    texts = []
    for prompt, completion in zip(examples["enhanced_prompt"], examples["enhanced_completion"]):
        if not prompt or not completion:
            texts.append("")
            continue
        texts.append(make_chat(prompt, completion))
    return {"text": texts}
ds_front = ds_front.map(fmt_front, batched=True)
ds_front = ds_front.filter(lambda x: len(x["text"]) > 100)
ds_front = ds_front.remove_columns([c for c in ds_front.column_names if c != "text"])
print(f"  → {len(ds_front)} примеров")

# --- 3. xlelords/vulcan ---
print("Загрузка xlelords/vulcan...")
ds_vulcan = load_dataset("xlelords/vulcan", split="train")
def fmt_vulcan(examples):
    texts = []
    for msgs in examples["messages"]:
        text = ""
        for m in msgs:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        texts.append(text.strip())
    return {"text": texts}
ds_vulcan = ds_vulcan.map(fmt_vulcan, batched=True)
ds_vulcan = ds_vulcan.filter(lambda x: len(x["text"]) > 100)
ds_vulcan = ds_vulcan.remove_columns([c for c in ds_vulcan.column_names if c != "text"])
print(f"  → {len(ds_vulcan)} примеров")

# --- 4. HuggingFaceTB/everyday-conversations (streaming, 1500) ---
print("Загрузка everyday-conversations (streaming)...")
ds_conv = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft", streaming=True)
conv_texts = []
for i, ex in enumerate(ds_conv):
    if i >= 1500:
        break
    msgs = ex.get("messages", [])
    if not msgs:
        continue
    text = ""
    for m in msgs:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    conv_texts.append({"text": text.strip()})
print(f"  → {len(conv_texts)} примеров")
ds_conv = Dataset.from_list(conv_texts)
ds_conv = ds_conv.filter(lambda x: len(x["text"]) > 100)

# --- 5. Ваш dataset.jsonl ---
print("Загрузка dataset.jsonl...")
with open("dataset.jsonl", "r", encoding="utf-8") as f:
    local_lines = f.readlines()
local_texts = []
for line in local_lines:
    obj = json.loads(line)
    inst = obj.get("instruction", "")
    inp = obj.get("input", "")
    out = obj.get("output", "")
    if not inst or not out:
        continue
    task = f"{inst}\n{inp}".strip() if inp else inst
    local_texts.append({"text": make_chat(task, out)})
ds_local = Dataset.from_list(local_texts)
ds_local = ds_local.filter(lambda x: len(x["text"]) > 100)
print(f"  → {len(ds_local)} примеров")

# --- 6. PuneetK/ShareGPT4-Instruction-Clean (streaming, 1000) ---
print("Загрузка ShareGPT4-Instruction-Clean (streaming)...")
try:
    ds_share = load_dataset("PuneetK/ShareGPT4-Instruction-Clean", split="train", streaming=True)
    share_texts = []
    for i, ex in enumerate(ds_share):
        if i >= 1000:
            break
        convs = ex.get("conversations", [])
        if not convs:
            continue
        text = ""
        for c in convs:
            role = "user" if c.get("from") == "human" else "assistant"
            text += f"<|im_start|>{role}\n{c.get('value', '')}<|im_end|>\n"
        share_texts.append({"text": text.strip()})
    print(f"  → {len(share_texts)} примеров")
    ds_share = Dataset.from_list(share_texts)
    ds_share = ds_share.filter(lambda x: len(x["text"]) > 100)
except Exception as e:
    print(f"  ⚠️  Не удалось загрузить: {str(e)[:150]}")
    ds_share = None

# --- ОБЪЕДИНЕНИЕ ---
print("\nОбъединение датасетов...")
to_concat = [ds_techta, ds_front, ds_vulcan, ds_conv, ds_local]
if ds_share is not None:
    to_concat.append(ds_share)
dataset = concatenate_datasets(to_concat)
dataset = dataset.shuffle(seed=3407)
print(f"Итого: {len(dataset)} примеров")

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
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        save_steps=500,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

print("Запуск обучения Qataclism Lotus 1.0...")
trainer.train()

# --- СОХРАНЕНИЕ LoRA-АДАПТЕРА ---
print("Сохранение адаптера...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Адаптер сохранён в {OUTPUT_DIR}")

# --- ЭКСПОРТ В GGUF ---
print("Экспорт Qataclism Lotus 1.0 в GGUF...")
model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method="q4_k_m"
)
print(f"Готово! GGUF сохранён в {GGUF_DIR}")