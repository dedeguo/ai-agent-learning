# 第三周：Transformer Block

目标：沿着张量形状追踪一个 Block，解释各组件如何协作。每课约 60～90 分钟，先预测、再运行、再解释。

| 课次 | 内容 | 讲解 |
| --- | --- | --- |
| 01 | 多头投影、拼接与输出投影 | [多头注意力](01_multi_head_attention.md) |
| 02 | 顺序与位置表示 | [位置信息](02_positions.md) |
| 03 | FFN、残差与 LayerNorm | [表示加工](03_ffn_residual_layernorm.md) |
| 04 | 完整 Pre-LN Block 与 PyTorch 对照 | [Block 数据流](04_transformer_block.md) |

Notebook 在 `notebooks/week03/`；从项目根目录运行 `.venv/bin/python exercises/week03/01_multi_head_attention.py`，其他课同理。

本周使用随机固定参数演示前向计算，无训练、dropout 或 causal mask。PyTorch 示例供阅读，运行需另外安装 torch；全部 NumPy 实验可用现有环境运行。

开始前自查：X 为 `(3, 4)`，Wq、Wk 为 `(4, 2)`，那么 QKᵀ 为什么是 `(3, 3)`？每行 softmax 在哪些位置之间分配权重？最后乘 V 为什么能汇集信息？

验收：画出 Block，标注每步形状；解释多头拼接与输出投影、位置信息、残差和 LayerNorm；从干净内核运行四份 Notebook。记录在 [学习进度](../../progress/week03.md)。
