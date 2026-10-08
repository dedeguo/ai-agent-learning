"""固定参数的 NumPy 教学 GPT；只做前向，不训练。"""
import numpy as np

VOCAB = ('我', '喜欢', '学习', 'AI', '<eos>', '你')


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def attention(x, causal=True):
    n, d = x.shape
    rng = np.random.default_rng(3)
    outputs, weights = [], []
    for _ in range(2):
        wq, wk, wv = [rng.normal(0, .4, (d, d // 2)) for _ in range(3)]
        q, k, v = x @ wq, x @ wk, x @ wv
        scores = q @ k.T / np.sqrt(d // 2)
        if causal:
            future = np.triu(np.ones((n, n), dtype=bool), k=1)
            scores = np.where(future, -np.inf, scores)
        a = softmax(scores)
        outputs.append(a @ v)
        weights.append(a)
    wo = rng.normal(0, .4, (d, d))
    return np.concatenate(outputs, axis=-1) @ wo, np.stack(weights)


def layer_norm(x):
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-5)


def gpt(ids):
    ids = np.asarray(ids, dtype=int)
    if ids.ndim != 1 or len(ids) == 0 or np.any((ids < 0) | (ids >= len(VOCAB))):
        raise ValueError('ids 必须是非空的一维有效 token ID 序列')
    rng = np.random.default_rng(10)
    d = 4
    embedding = rng.normal(0, .4, (len(VOCAB), d))
    angles = np.arange(len(ids))[:, None] * 10000. ** (-np.arange(0, d, 2) / d)
    positions = np.zeros((len(ids), d))
    positions[:, 0::2], positions[:, 1::2] = np.sin(angles), np.cos(angles)
    x = embedding[ids] + positions
    for _ in range(2):
        # 两个 Block 使用不同 FFN 参数；本教学例共享 Attention 参数。
        a, _ = attention(layer_norm(x))
        x = x + a
        w1 = rng.normal(0, .4, (d, 2*d))
        w2 = rng.normal(0, .4, (2*d, d))
        x = x + np.maximum(layer_norm(x) @ w1, 0) @ w2
    output_projection = rng.normal(0, .4, (d, len(VOCAB)))
    return layer_norm(x) @ output_projection


def cross_entropy(logits, targets):
    # 使用 log-sum-exp，避免先转概率再取 log 时下溢。
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return -log_probs[np.arange(len(targets)), targets]


def sample(logits, temperature, rng):
    if temperature < 0:
        raise ValueError('temperature 不能为负数')
    if temperature == 0:
        return int(np.argmax(logits))
    return int(rng.choice(len(logits), p=softmax(logits / temperature)))
