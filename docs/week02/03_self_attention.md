# 第三课：逐步计算 Self-Attention

```text
scores = Q @ K.T / sqrt(d_k)
A = softmax(scores, 按行)
O = A @ V
```

## 1. 为什么转置 K

Q 为 `(n, d_k)`，K.T 为 `(d_k, n)`，结果为 `(n, n)`。其中 `[i, j]` 是查询位置 i 与键位置 j 的点积。每一行回答：当前位置应从哪些位置获取信息？包括它自己。

直接 `Q @ K` 通常形状不匹配；即使尺寸碰巧能乘，也不是在计算这些两两点积。

## 2. 为什么缩放

在各分量近似独立、零均值且方差约为 1 的假设下，d_k 项点积之和的方差约为 d_k。除以 `sqrt(d_k)` 有助于稳定分数量级，减轻高维时 softmax 过于尖锐的问题。这是设计动机，不保证任意输入的方差都精确为 1。

## 3. 为什么按行 softmax

每个查询位置对所有键位置生成一组权重：

```python
shifted = scores - scores.max(axis=1, keepdims=True)
exp_scores = np.exp(shifted)
A = exp_scores / exp_scores.sum(axis=1, keepdims=True)
```

减去每行最大值提高数值稳定性，不改变该行 softmax。`A[i, j]` 表示输出位置 i 给 V 的第 j 行多少权重。每行之和为 1，列和一般不是 1；A 也不一定对称。

## 4. 为什么最后乘 V

`(n, n) @ (n, d_v) → (n, d_v)`，即：

```text
O[i] = A[i, 0] × V[0] + … + A[i, n-1] × V[n-1]
```

例如权重为 `[0.2, 0.3, 0.5]`，V 的三行为 `[1, 0]`、`[0, 1]`、`[2, 2]`，输出为 `[1.2, 1.3]`。输出是向量表示，不是最终词表概率。

运行 `03_self_attention`，依次查看 Q、K、V、原始分数、缩放分数、A 与 O。Notebook 会用热力图展示 A；纵轴是 query 位置，横轴是 key 位置。

本例每个位置可以看见全部位置，未加 causal mask，因此不是完整 GPT 的注意力计算。

自查：若 n=4、d_k=2、d_v=3，分数与输出分别是什么形状？答案：`(4, 4)` 与 `(4, 3)`。

下一课：[修改输入做实验](04_attention_experiments.md)。
