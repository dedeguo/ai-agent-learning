import numpy as np
from attention_utils import embed, attention, show

_, _, X = embed("我 喜欢 学习")
result = attention(X)
for name, value in result.items():
    show(name, value)
print("每行权重之和:", result["A"].sum(axis=1))
np.testing.assert_allclose(result["A"].sum(axis=1), np.ones(len(X)))
manual_first = sum(result["A"][0, j] * result["V"][j] for j in range(len(X)))
np.testing.assert_allclose(manual_first, result["O"][0])
print("第一行加权求和核对通过:", manual_first)
