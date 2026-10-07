from attention_utils import embed, show

text = "我 喜欢 学习"  # 修改这里，再运行
tokens, ids, X = embed(text)
print("tokens:", tokens)
show("ids", ids)
show("X", X)
_, _, repeated = embed("我 喜欢 我")
print("重复 token 的输入向量相同:", (repeated[0] == repeated[2]).all())
