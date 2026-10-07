"""固定随机参数的教学前向计算；不包含训练。"""
import numpy as np

X = np.array([[1., 0., 0., 1.], [0., 1., 0., 1.], [1., 1., 0., 0.]])

def layer_norm(x, gamma=1., beta=0., eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def positions(n, d_model):
    pos = np.arange(n)[:, None]
    rates = 10000. ** (-np.arange(0, d_model, 2) / d_model)
    angles = pos * rates
    out = np.zeros((n, d_model))
    out[:, 0::2] = np.sin(angles)
    out[:, 1::2] = np.cos(angles[:, :out[:, 1::2].shape[1]])
    return out

def multi_head(x, heads=2):
    n, d = x.shape
    if heads <= 0 or d % heads:
        raise ValueError("heads 必须为正整数且整除 d_model")
    dh = d // heads
    rng = np.random.default_rng(3)
    outputs, weights = [], []
    for _ in range(heads):
        wq, wk, wv = [rng.normal(0, .4, (d, dh)) for _ in range(3)]
        q, k, v = x @ wq, x @ wk, x @ wv
        scores = q @ k.T / np.sqrt(dh)
        exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        a = exp / exp.sum(axis=-1, keepdims=True)
        weights.append(a)
        outputs.append(a @ v)
    concat = np.concatenate(outputs, axis=-1)
    wo = rng.normal(0, .4, (d, d))
    return concat @ wo, np.stack(weights), concat

def ffn(x):
    rng = np.random.default_rng(7)
    d = x.shape[-1]
    w1 = rng.normal(0, .4, (d, 2*d))
    w2 = rng.normal(0, .4, (2*d, d))
    b1 = np.zeros(2*d)
    b2 = np.zeros(d)
    return np.maximum(x @ w1 + b1, 0) @ w2 + b2

def block(x):
    normalized = layer_norm(x)
    attn, weights, _ = multi_head(normalized)
    h = x + attn
    ff = ffn(layer_norm(h))
    return dict(X=x, normalized=normalized, attention=attn,
                weights=weights, H=h, ffn=ff, Y=h+ff)

def show(name, value):
    print(f"{name} shape={value.shape}\n{value}\n")
