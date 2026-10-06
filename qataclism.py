
import json
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

# --- КОНФИГУРАЦИЯ QATACLISM 1.0 ROSE 7B ---
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./Qataclism-1.0-Rose-7B"
GGUF_DIR = "Qataclism-1.0-Rose-7B-GGUF"
MAX_SEQ_LENGTH = 1024
SYSTEM_PROMPT = "You are Qataclism 1.0 Rose 7B, a versatile AI assistant skilled in backend development, security analysis, and multilingual translation."

# --- ЗАГРУЗКА МОДЕЛИ ---
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
        texts.append(make_chat(task, "\n\n".join(code_parts)))
    return {"text": texts}

ds_techta = ds_techta.map(fmt_techta, batched=True)
ds_techta = ds_techta.filter(lambda x: len(x["text"]) > 100)
ds_techta = ds_techta.remove_columns([c for c in ds_techta.column_names if c != "text"])
print(f"  → {len(ds_techta)} примеров")

# --- 2. matteopilotto/rust-github-issues ---
print("Загрузка matteopilotto/rust-github-issues...")
ds_rust = load_dataset("matteopilotto/rust-github-issues", split="train")

def fmt_rust(examples):
    texts = []
    for title, body, comments in zip(examples["title"], examples["body"], examples["comments"]):
        task = f"GitHub issue:\nTitle: {title}\n\nBody:\n{body}"
        texts.append(make_chat(task, comments or ""))
    return {"text": texts}

ds_rust = ds_rust.map(fmt_rust, batched=True)
ds_rust = ds_rust.filter(lambda x: len(x["text"]) > 100)
ds_rust = ds_rust.remove_columns([c for c in ds_rust.column_names if c != "text"])
print(f"  → {len(ds_rust)} примеров")

# --- 3. relai-ai/flask-standard ---
print("Загрузка relai-ai/flask-standard...")
ds_flask = load_dataset("relai-ai/flask-standard", split="train")

def fmt_flask(examples):
    texts = []
    for q, r in zip(examples["Question"], examples["Response"]):
        texts.append(make_chat(q, r))
    return {"text": texts}

ds_flask = ds_flask.map(fmt_flask, batched=True)
ds_flask = ds_flask.remove_columns([c for c in ds_flask.column_names if c != "text"])
print(f"  → {len(ds_flask)} примеров")

# --- 4. l1ghth4t/iast-python3-django-flask ---
print("Загрузка l1ghth4t/iast-python3-django-flask...")
ds_iast = load_dataset("l1ghth4t/iast-python3-django-flask", split="train")

def fmt_iast(examples):
    texts = []
    for text_dict, label in zip(examples["text"], examples["label"]):
        try:
            payload = json.dumps(text_dict, ensure_ascii=False, indent=2)
        except Exception:
            payload = str(text_dict)
        task = f"Analyze this HTTP request for security vulnerabilities:\n{payload}"
        assistant = f"Detected vulnerability class: {label}"
        texts.append(make_chat(task, assistant))
    return {"text": texts}

ds_iast = ds_iast.map(fmt_iast, batched=True)
ds_iast = ds_iast.filter(lambda x: len(x["text"]) > 100)
ds_iast = ds_iast.remove_columns([c for c in ds_iast.column_names if c != "text"])
print(f"  → {len(ds_iast)} примеров")

# --- 5. Helsinki-NLP/opus_books (en-ru) ---
print("Загрузка Helsinki-NLP/opus_books (en-ru)...")
ds_books = load_dataset("Helsinki-NLP/opus_books", "en-ru", split="train")

def fmt_books(examples):
    texts = []
    for tr in examples["translation"]:
        en = tr.get("en", "")
        ru = tr.get("ru", "")
        if not en or not ru:
            texts.append("")
            continue
        task = f"Translate from English to Russian:\n{en}"
        texts.append(make_chat(task, ru))
    return {"text": texts}

ds_books = ds_books.map(fmt_books, batched=True)
ds_books = ds_books.filter(lambda x: len(x["text"]) > 20)
ds_books = ds_books.remove_columns([c for c in ds_books.column_names if c != "text"])
print(f"  → {len(ds_books)} примеров")

# --- ОБЪЕДИНЕНИЕ ---
print("\nОбъединение датасетов...")
dataset = concatenate_datasets([ds_techta, ds_rust, ds_flask, ds_iast, ds_books])
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
        gradient_accumulation_steps=8,
        warmup_steps=20,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=200,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

print("Запуск обучения Qataclism 1.0 Rose 7B...")
trainer.train()

# --- СОХРАНЕНИЕ В GGUF ---
print("Экспорт Qataclism 1.0 Rose 7B в GGUF...")
model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method="q4_k_m"
)
print(f"Готово! Модель сохранена в {GGUF_DIR}")