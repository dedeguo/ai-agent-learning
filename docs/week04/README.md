# 第四周：GPT、训练与生成

目标：把第三周的 Block 连成 GPT，理解模型如何预测下一 token，以及训练和生成有什么区别。每课约 60～90 分钟。

| 课次 | 内容 | 讲义 |
| --- | --- | --- |
| 01 | decoder-only 与 causal mask | [GPT 的信息流](01_gpt_and_causal_mask.md) |
| 02 | 输入和标签错开一位、交叉熵 | [下一 token 预测](02_next_token_prediction.md) |
| 03 | 自回归、temperature 与采样 | [逐步生成](03_autoregressive_sampling.md) |
| 04 | 预训练、SFT、RLHF、DPO 与推理时计算 | [训练阶段与推理](04_training_and_reasoning.md) |

先读讲义，在 `notebooks/week04/` 运行对应 Notebook，也可以从项目根目录运行 `.venv/bin/python exercises/week04/01_gpt_and_causal_mask.py`，其他课同理。

本周只依赖 NumPy。示例采用固定随机参数、玩具词表，无参数训练；输出用来观察计算机制，不能用来判断语言能力。可选 mini GPT 训练不属于本周必做任务。

开始前自查：能否解释 `QKᵀ` 的形状、每行 softmax 的含义、残差连接和 FFN？有疑问时回看第三周对应讲义。

验收：画出 GPT 流程；说明 mask 在 softmax 前生效；构造错位标签；解释训练并行与生成串行；比较 temperature、SFT 和偏好优化的作用。记录在 [学习进度](../../progress/week04.md)。
