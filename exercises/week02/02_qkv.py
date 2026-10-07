from attention_utils import embed, attention, show, Wq, Wk, Wv

_, _, X = embed("我 喜欢 学习")
result = attention(X)
for name, value in [("Wq", Wq), ("Wk", Wk), ("Wv", Wv)]:
    show(name, value)
for name in ["X", "Q", "K", "V","O"]:
    show(name, result[name])
