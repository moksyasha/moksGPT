"""этап 1 — bigram baseline. предсказываем след. символ по одному текущему.
цель не качество, а поднять весь train-loop: модель -> loss -> backward -> generate."""
import torch
import torch.nn as nn
from torch.nn import functional as F

# переиспользуем всё из этапа 0 (data.py при импорте читает текст и строит vocab)
from data import vocab_size, get_batch, encode, decode

# --- гиперпараметры ---
device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 8       # длина куска текста в батче (для bigram роли не играет, нужен для get_batch)
batch_size = 32
steps = 3000
lr = 1e-2            # bigram учится быстро, можно крупный шаг
torch.manual_seed(1337)


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()  # обязателен: инициализирует машинерию nn.Module
        # таблица vocab x vocab: строка для символа i = logits след. символа
        # 1 арг это num_embeddings, размер словаря сколько строк в таблице W, какие макс id доступны на вход лонг числа, 
        # 2 арг выход - длина вектора, в который распаковывается каждый id на входе
        self.token_emb = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) — id символов. lookup в таблице -> (B, T, vocab) logits
        logits = self.token_emb(idx)

        if targets is None:
            return logits, None
        # logits размером batch * block_size * vocab_size
        # targets размером batch * block_size
        B, T, V = logits.shape
        # >>> YOU DO (дырка 1) <<<
        # F.cross_entropy хочет logits формы (N, V) и targets формы (N,), где N = B*T.
        # расплющи обе оси B и T в одну. подсказка: .view(...) / .reshape(...).
        # logits сейчас (B, T, V), targets сейчас (B, T).
        logits_flat = logits.view(-1, V)   # -> (B*T, V)
        targets_flat = targets.view(-1,)  # -> (B*T,)
        loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    @torch.no_grad()  # при генерации градиенты не нужны
    def generate(self, idx, max_new_tokens):
        # idx: (B, T) — стартовый контекст. дописываем max_new_tokens символов
        for _ in range(max_new_tokens):
            logits, _ = self(idx)            # (B, T, vocab)
            logits = logits[:, -1, :]        # берём только последнюю позицию -> (B, vocab)
            probs = F.softmax(logits, dim=-1)  # logits -> вероятности
            next_id = torch.multinomial(probs, num_samples=1)  # сэмплим (B, 1)
            idx = torch.cat([idx, next_id], dim=1)  # дописываем -> (B, T+1)
        return idx


model = BigramModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# --- цикл обучения ---
for step in range(steps):
    xb, yb = get_batch("train", block_size, batch_size, device)
    logits, loss = model(xb, yb)

    # >>> YOU DO (дырка 2) <<<
    # три шага autograd в правильном порядке: обнулить градиенты, посчитать
    # градиент loss по весам, шагнуть оптимизатором.
    # 1. Обнуляем старые градиенты, чтобы они не накапливались
    optimizer.zero_grad()
    # 2. Обратное распространение (backward pass): вычисляем новые градиенты
    loss.backward()
    # 3. Шагаем оптимизатором
    optimizer.step()

    if step % 500 == 0:
        print(f"step {step:4d} | loss {loss.item():.4f}")

# --- генерация ---
start = torch.zeros((1, 1), dtype=torch.long, device=device)  # начинаем с символа id=0 ('\n')
out = model.generate(start, max_new_tokens=300)[0].tolist()
print("\n--- sample ---")
print(decode(out))
