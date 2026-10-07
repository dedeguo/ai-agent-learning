# 第二周：Token、Embedding 与 Self-Attention

目标：从一句短文本出发，逐步算出 Attention，解释各个矩阵的含义与形状。

| 课次 | 内容 | 文档与实验 |
| --- | --- | --- |
| 01 | Token、ID、Embedding 查表 | [讲解](01_tokens_and_embeddings.md) · `01_tokens_and_embeddings` |
| 02 | 从 X 投影到 Q、K、V | [讲解](02_qkv.md) · `02_qkv` |
| 03 | 分数、缩放、按行 softmax、加权求和 | [讲解](03_self_attention.md) · `03_self_attention` |
| 04 | 修改输入，观察权重与输出 | [讲解](04_attention_experiments.md) · `04_attention_experiments` |

每课约 60～90 分钟：先预测结果 → 阅读讲解 → 运行 Notebook → 修改输入 → 记录解释。

从第一课开始，Notebook 位于 `notebooks/week02/`。独立脚本在项目根目录运行：

```bash
.venv/bin/python exercises/week02/01_tokens_and_embeddings.py
.venv/bin/python exercises/week02/02_qkv.py
.venv/bin/python exercises/week02/03_self_attention.py
.venv/bin/python exercises/week02/04_attention_experiments.py
```

进入 QKV 前的小检查：`(3, 4) @ (4, 2)` 的结果是什么形状？`(3, 2) @ (2, 3)` 呢？答案分别是 `(3, 2)` 和 `(3, 3)`。不熟悉时回看第一周矩阵乘法与按行 softmax。

本周使用手工词表和参数，只演示前向计算，不包含模型训练、位置信息或 causal mask；后两项分别在第三、四周展开。

验收：独立解释为何乘 `K.T`、为何除以 `sqrt(d_k)`、为何最后乘 `V`；检查每行注意力权重之和为 1；能修改一个输入并解释输出变化。记录在 [第二周进度](../../progress/week02.md)。

## 图解与交互实验

打开 [单头 Self-Attention 详细图解](self_attention_explained.html)，逐步查看手算、矩阵与交互实验；[独立流程图](assets/single-head-attention.svg) 可保存或插入笔记。HTML 页面可离线打开。
