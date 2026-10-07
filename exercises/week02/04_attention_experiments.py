import numpy as np
from attention_utils import embed, attention, softmax_rows, show

_, _, X = embed("我 喜欢 学习")
before = attention(X)
changed_X = X.copy()
changed_X[2, 0] += 1
changed = attention(changed_X)
for name in ["Q", "K", "V", "scores", "A", "O"]:
    show(name + " 差值", changed[name] - before[name])

# 隔离 V：不重新投影 Q、K。
changed_V = before["V"].copy()
changed_V[2, 0] += 1
show("只修改 V 时的输出差值", before["A"] @ changed_V - before["O"])

# 隔离 Q：只改变查询位置 0。
changed_Q = before["Q"].copy()
changed_Q[0, 0] += 1
query_A = softmax_rows(changed_Q @ before["K"].T / np.sqrt(changed_Q.shape[1]))
show("只修改 Q 时的权重差值", query_A - before["A"])
np.testing.assert_allclose(query_A[1:], before["A"][1:])

# 每行加不同的常数，行内相同：softmax 不变。
shift = np.array([[3.0], [-2.0], [8.0]])
np.testing.assert_allclose(softmax_rows(before["scores"] + shift), before["A"])
print("对照实验核对通过")
