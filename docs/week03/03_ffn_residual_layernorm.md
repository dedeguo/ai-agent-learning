# 第三课：FFN、残差和 LayerNorm

Attention 在不同 token 之间交换信息；FFN 用同一套参数，分别加工每个 token 的特征：

```text
FFN(X) = ReLU(X @ W1 + b1) @ W2 + b2
(n,4) → (n,8) → (n,4)
```

中间维度在本例扩大到 8；实际配置可以不同，也可以使用 GELU 或门控结构。FFN 的矩阵乘法不混合 token 行，但输入表示已经可能通过 Attention 包含其他 token 的信息。

残差：`Y = X + F(X)`。形状必须兼容；保留一条直接路径，让模块学习对表示的增量，帮助梯度传播。它不保证每次输出都更好，也不保证训练必然稳定。

LayerNorm 对**每一个 token 内的特征维**求均值与方差：

```text
LN(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
```

gamma、beta 是可学习参数。它不是按列跨 token 归一化，也不是把向量变成概率。只有在 gamma=1、beta=0 时，输出每行均值约为 0、方差约为 1；eps 会使方差略低于 1。

运行 `03_ffn_residual_layernorm`：查看每行均值和方差；只修改 X 的第一行，验证 FFN 的其余行不变。尝试修改 gamma 或 beta，再检查均值是否仍为 0。

自查：FFN 中间维度扩大后，为什么还要投影回 d_model？残差相加与拼接分别改变什么形状？
