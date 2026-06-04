"""этап 0 — токенизация и батч-сэмплер (char-level, tinyshakespeare)."""
import torch

# --- 1. читаем текст ---
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# --- 2. строим vocab: все уникальные символы, отсортированы для стабильности ---
chars = sorted(set(text))
vocab_size = len(chars)

# словари в обе стороны: символ -> id и id -> символ
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# encode: строка -> список id;  decode: список id -> строка
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

# --- 3. весь текст -> один длинный тензор id, делим на train/val ---
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# --- 4. батч-сэмплер ---
def get_batch(split, block_size, batch_size, device="cpu"):
    d = train_data if split == "train" else val_data
    # случайные стартовые позиции, по одной на каждый элемент батча randint(low=0, high, size
    ix = torch.randint(len(d) - block_size, (batch_size,))
    # x: куски длины block_size; y: те же куски, сдвинутые на 1 вперёд
    x = torch.stack([d[i : i + block_size] for i in ix])
    # >>> YOU DO <<<
    # собери y по аналогии с x, но сдвинутый на 1 символ вперёд.
    # подсказка: тот же torch.stack и тот же ix, измени только индексы среза.
    y = torch.stack([d[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    print(f"длина текста: {len(text):,} символов")
    print(f"vocab_size: {vocab_size}")
    print(f"vocab: {''.join(chars)!r}")
    print(f"encode('hi') = {encode('hi')}  decode обратно = {decode(encode('hi'))!r}")

    xb, yb = get_batch("train", block_size=8, batch_size=4)
    print(f"\nx shape: {xb.shape}, y shape: {yb.shape}")
    print("x[0]:", xb[0].tolist())
    print("y[0]:", yb[0].tolist())
    # проверка: y[0] должен быть x[0], сдвинутый на 1 (y[0][k] == x[0][k+1])
    assert xb[0][1] == yb[0][0]