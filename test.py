
from datasets import load_dataset

# Список датасетов для проверки
DATASETS = [
    "wgyhhh/RealTimeNews-2025",
    "Techta/backend-code-generator-dataset",
    "matteopilotto/rust-github-issues",
    "relai-ai/flask-standard",
    "l1ghth4t/iast-python3-django-flask",
    "Helsinki-NLP/opus_books",
]

for name in DATASETS:
    print("=" * 80)
    print(f"📦 Датасет: {name}")
    print("=" * 80)

    try:
        # Загружаем только train split (или первый доступный)
        ds = load_dataset(name, split="train")
    except Exception as e:
        print(f"⚠️  Не удалось загрузить 'train': {e}")
        try:
            # Пробуем без указания split
            ds = load_dataset(name)
            print(f"Доступные splits: {list(ds.keys())}")
            # Берём первый попавшийся
            first_split = list(ds.keys())[0]
            ds = ds[first_split]
            print(f"Использую split: {first_split}")
        except Exception as e2:
            print(f"❌ Полностью не удалось загрузить: {e2}")
            print()
            continue

    # Размер
    print(f"📊 Размер (строк): {len(ds):,}")

    # Колонки
    print(f"📁 Колонки: {ds.column_names}")

    # Пример первой строки
    print(f"\n🔍 Пример первого элемента:")
    first = ds[0]
    for col in ds.column_names:
        value = first[col]
        # Обрезаем длинные значения для читаемости
        if isinstance(value, str) and len(value) > 300:
            value = value[:300] + "... [обрезано]"
        elif isinstance(value, (list, dict)):
            value = str(value)[:300] + "... [обрезано]"
        print(f"   • {col}: {value}")

    print()