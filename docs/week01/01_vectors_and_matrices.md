# 第一课：标量、向量与矩阵

目标：理解基本数学对象，判断矩阵是否可以相乘，并计算点积、向量的模与余弦相似度。

## 1. 标量、向量、矩阵与形状

| 对象 | 含义 | 示例 | NumPy 形状 |
| --- | --- | --- | --- |
| 标量 | 一个数字 | `3.0` | 用零维数组表示时为 `()` |
| 向量 | 有顺序的一组数字 | `[80, 90, 70]` | 一维数组为 `(3,)` |
| 矩阵 | 按行列排列的数字 | 两个学生、每人三门成绩 | `(2, 3)` |

```python
import numpy as np

score = 3.0  # Python 浮点数，本身没有 .shape 属性
student = np.array([80, 90, 70])
scores = np.array([
    [80, 90, 70],  # 学生 1
    [60, 75, 85],  # 学生 2
])

print(np.array(score).shape)  # ()
print(student.shape)         # (3,)
print(scores.shape)          # (2, 3)
```

这里每行代表一个学生，每列代表一门课。形状告诉我们数据如何排列，但每个轴的实际含义需要由任务约定。

### 一维向量与列矩阵不同

```python
vector = np.array([1, 2, 3])
column = np.array([[1], [2], [3]])
row = np.array([[1, 2, 3]])

print(vector.shape)  # (3,)：一维数组，没有独立的行轴和列轴
print(column.shape)  # (3, 1)：三行一列
print(row.shape)     # (1, 3)：一行三列
```

数学里的三维向量在 NumPy 中可以用这些不同形状表示。不要把向量的分量数、数组的轴数与向量的长度混淆：`[1, 2, 3]` 有三个分量，但 NumPy 数组只有一个轴；长度（模）是另一个概念。

## 2. 转置：交换行和列

矩阵转置把原来的行变成列、列变成行。用 `.T` 表示：

```python
print(scores.T)
# [[80, 60],
#  [90, 75],
#  [70, 85]]
print(scores.T.shape)  # (3, 2)
```

原矩阵每行是一个学生的三门成绩；转置后每行是一门课中两个学生的成绩，每列是一个学生的三门成绩。

二维形状 `(m, n)` 转置后为 `(n, m)`。一维数组的 `.T` 不会变成列矩阵：

```python
print(vector.T.shape)               # 仍是 (3,)
print(vector.reshape(-1, 1).shape)   # (3, 1)：显式转换为列矩阵
```

## 3. 点积：对应相乘再求和

两个长度相同的一维向量的点积是一个标量：

```text
[2, 3, 4] · [1, 0, 2]
= 2×1 + 3×0 + 4×2
= 10
```

```python
x = np.array([2, 3, 4])
w = np.array([1, 0, 2])
print(x @ w)           # 10
print(np.dot(x, w))    # 10，对于这里的一维向量，两种写法等价
```

例如，用权重 `[0.5, 0.3, 0.2]` 计算综合成绩，就是成绩与权重的点积：

```text
80×0.5 + 90×0.3 + 70×0.2 = 81
```

点积既与向量长度有关，也与方向有关，不能直接等同于只比较方向的相似度。

## 4. 矩阵乘法：每行与每列做点积

规则：

```text
(m, n) @ (n, p) → (m, p)
```

中间维度必须相同，结果取外侧维度。结果第 i 行、第 j 列的元素，是左矩阵第 i 行与右矩阵第 j 列的点积。

| 输入形状 | 是否可以相乘 | 输出形状 |
| --- | --- | --- |
| `(4, 3) @ (3, 2)` | 可以 | `(4, 2)` |
| `(2, 3) @ (3, 4)` | 可以 | `(2, 4)` |
| `(2, 3) @ (2, 4)` | 不可以 | 中间维度不同 |

### 一次计算多个学生的综合成绩

```python
weights = np.array([0.5, 0.3, 0.2])
overall = scores @ weights
print(overall)        # [81., 69.5]
print(overall.shape)  # (2,)

weights_column = weights.reshape(-1, 1)
overall_column = scores @ weights_column
print(overall_column)        # [[81.], [69.5]]
print(overall_column.shape)  # (2, 1)
```

两次计算的成绩相同，结果形状不同：

```text
(2, 3) @ (3,)   → (2,)
(2, 3) @ (3, 1) → (2, 1)
```

本次练习曾将第一种结果判断为 `(2, 1)`，需要注意 NumPy 一维数组与列矩阵的区别。

### scores @ scores.T

```python
print(scores @ scores.T)
# [[19400, 17500],
#  [17500, 16450]]
```

形状为 `(2, 3) @ (3, 2) → (2, 2)`。第 i 行、第 j 列是学生 i 与学生 j 的成绩向量点积。例如第一行第二列是 `80×60 + 90×75 + 70×85 = 17500`。

### @ 与 * 不同

`@` 表示矩阵乘法（两个一维向量时得到点积）；`*` 表示逐元素乘法，并可能触发广播：

```python
print(x * w)  # [2, 0, 8]，保留各位置乘积
print(x @ w)  # 10，将对应乘积求和
```

矩阵乘法通常不满足交换律：`A @ B` 与 `B @ A` 不一定相同，反过来也可能无法相乘。

## 5. 向量的模：长度不是元素个数

本课使用欧几里得模（L2 范数）：

```text
‖a‖ = √(a₁² + a₂² + … + aₙ²)
```

```python
a = np.array([1.0, 2.0])
b = np.array([2.0, 4.0])

print(np.linalg.norm(a))          # √5 ≈ 2.2361
print(np.sqrt(np.sum(a ** 2)))    # 相同结果
print(np.linalg.norm(b))          # 2√5 ≈ 4.4721
```

两个向量都有两个分量，但 b 是 a 的两倍：方向相同，模不同。零向量的模为 0。

## 6. 余弦相似度：比较非零向量的方向

```text
cos(a, b) = (a · b) / (‖a‖ × ‖b‖)
```

用两个模的乘积归一化点积，消除长度对这一比较的影响。对于非零实向量，结果范围为 [-1, 1]：

| 结果 | 几何含义 |
| --- | --- |
| 1 | 方向相同 |
| 0 | 相互垂直（正交） |
| -1 | 方向相反 |

```python
def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("零向量的余弦相似度未定义")
    return (a @ b) / (norm_a * norm_b)

print(cosine_similarity([1, 2], [2, 4]))    # 约 1
print(cosine_similarity([1, 0], [0, 1]))   # 0
print(cosine_similarity([1, 2], [-1, -2])) # 约 -1
```

函数用于本课长度相同的一维向量。零向量没有可确定的方向，因此公式不能直接应用。

### 分母括号是本次练习的易错点

```python
a = np.array([1.0, 2.0])
b = np.array([2.0, 4.0])

wrong = (a @ b) / np.linalg.norm(a) * np.linalg.norm(b)
correct = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(wrong)    # 约 20：不是余弦相似度
print(correct)  # 约 1
```

乘除按同级从左到右计算。错误写法先除以 a 的模，再乘以 b 的模；正确写法除以两个模的乘积。若 b 的模恰好为 1，两种写法碰巧相同，不能因此证明公式正确。

余弦相似度经常用于比较文本 embedding，但向量方向如何对应语义，取决于表示模型；相似度高不直接保证两段文字事实一致。

## 7. 速查与自查

| 操作 | NumPy 写法 |
| --- | --- |
| 查看形状 | `a.shape` |
| 二维矩阵转置 | `A.T` |
| 转成列矩阵 | `a.reshape(-1, 1)` |
| 点积 / 矩阵乘法 | `a @ b` / `A @ B` |
| 逐元素乘法 | `a * b` |
| 一维向量的欧几里得模 | `np.linalg.norm(a)` |
| 余弦相似度 | `(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))` |

自查：

1. `(4, 3) @ (3, 2)` 的结果形状是什么？
2. 为什么 `(3,)` 的 `.T` 不会变成 `(3, 1)`？
3. `[1, 2]` 与 `[2, 4]` 的模与方向有什么关系？
4. `[1, 0]` 与 `[0, 1]` 的点积和余弦相似度是多少？
5. 为什么余弦相似度的分母需要括号，为什么不能使用零向量？

对应实验：[第一课 Notebook](../../notebooks/week01/01_vectors_and_matrices.ipynb)。返回[第一周安排](README.md)。
