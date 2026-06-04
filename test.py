import torch

# --- 1. читаем текст ---
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = set(text)
vocab_size = len(chars)
print(f"длина текста: {len(text):,} символов {chars}, {len(chars)}")
print(f"размер словаря: {vocab_size}")

ix = torch.randint(len(chars) - 4, (2,))
print(f"случайные стартовые позиции: {ix}")

a = torch.stack([torch.arange(i, i + 4) for i in ix])
print(f"результат stack: {a}, {a.shape}")