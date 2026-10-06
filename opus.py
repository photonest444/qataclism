
from datasets import load_dataset

print("=" * 80)
print("📦 Helsinki-NLP/opus_books (en-ru)")
print("=" * 80)

ds = load_dataset("Helsinki-NLP/opus_books", "en-ru", split="train")
print(f"📊 Размер: {len(ds):,}")
print(f"📁 Колонки: {ds.column_names}")
print(f"\n🔍 Пример:")
for col in ds.column_names:
    val = ds[0][col]
    if isinstance(val, str) and len(val) > 300:
        val = val[:300] + "..."
    print(f"   • {col}: {val}")