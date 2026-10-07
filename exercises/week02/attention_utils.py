"""第二周玩具数据与单头 Self-Attention；参数手工指定，无训练。"""
import numpy as np

TOKENS = ["我", "喜欢", "学习", "你", "[UNK]"]
VOCAB = {token: i for i, token in enumerate(TOKENS)}
E = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1],
              [1, 1, 0, 0], [0, 0, 0, 0]], dtype=float)
Wq = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
Wk = np.array([[1, 0], [0, 1], [1, -1], [0, 0]], dtype=float)
Wv = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)


def embed(text):
    tokens = text.split()  # 仅供演示的空格分词
    if not tokens:
        raise ValueError("请输入至少一个 token")
    ids = np.array([VOCAB.get(t, VOCAB["[UNK]"]) for t in tokens])
    return tokens, ids, E[ids]


def softmax_rows(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def attention(X):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    raw = Q @ K.T
    scores = raw / np.sqrt(Q.shape[1])
    A = softmax_rows(scores)
    O = A @ V
    return dict(X=X, Q=Q, K=K, V=V, raw=raw, scores=scores, A=A, O=O)


def show(name, value):
    print(f"{name} shape={value.shape}\n{value}\n")
