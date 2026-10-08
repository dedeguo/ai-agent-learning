# 第二课：训练时如何预测下一 token

给定 token 序列 `[我, 喜欢, 学习, AI, <eos>]`，把输入与目标错开一位：

| 位置 | 输入 token | 目标 token |
| --- | --- | --- |
| 0 | 我 | 喜欢 |
| 1 | 喜欢 | 学习 |
| 2 | 学习 | AI |
| 3 | AI | <eos> |

每个位置输出一组词表 logits。causal mask 确保位置 0 不能从输入位置 1 偷看“喜欢”。训练时输入整段已知文本，称为 teacher forcing；不是把当前预测作为下一个训练位置的输入。

```text
inputs = ids[:-1]
targets = ids[1:]
logits.shape = (4, vocab_size)
loss_i = -log p(targets[i] | ids[:i+1])
loss = 四个位置 loss_i 的平均值
```

多个位置可在同一次前向中并行计算，因果约束仍然存在。真实 batch 常有 padding；padding 目标通常不计入 loss。SFT 还常只计算回答部分的 loss。本课不涉及这些额外掩码。

训练循环是“前向 → loss → 反向传播 → 更新参数”。本课只计算前向和 loss，没有梯度或参数更新。固定随机模型的 loss 不代表训练结果。若预测均匀，单个目标的 loss 为 `log(vocab_size)`。

运行第二课，检查输入/标签映射，再比较随机预测与人为提高正确目标 logits 后的 loss。后一实验只是验证交叉熵的作用，不等于模型学会了预测。

验收：为 `[A,B,C,D]` 写出输入和标签；解释为何训练能一次计算三个预测，而生成下一段文本必须逐步推进。
