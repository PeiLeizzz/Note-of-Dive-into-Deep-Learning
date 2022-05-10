## 预备知识

### 数据操作（PyTorch）

- 和 NumPy 的 API 基本差不多

- 原地更新参数：`x[:] = y` 或者 `x += y`

- 转换为其他对象

    ```python
    A = X.numpy() # torch -> numpy
    B = torch.tensor(A) # numpy -> torch
    
    a = torch.tensor([3.5])
    a.item() # -> value
    float(a) # -> float
    int(a) # -> int
    ```

### 数据预处理

- 处理缺失值：`.fillna() pd.get_dummies()`

### 线性代数

- 张量的维度用来表示张量具有的轴数，在这个意义上，张量的某个轴的维数就是这个轴的长度

- 将每个数据样本作为矩阵中的行向量更为常见。沿着张量的最外轴，我们可以访问或遍历小批量的数据样本；如果不存在小批量，我们也可以只访问数据样本

- 默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量，我们还可以指定张量沿哪一个轴来通过求和降低维度

- 在求和或均值时也可以保持轴数不变 `keepdims=True`

- 点积：`torch.dot(x, y)` 或 `torch.sum(x * y)`

- 矩阵-向量积：矩阵的每一行和向量做点积，所以矩阵的列维数必须与向量的长度相同 `torch.mv(A, x)`

    $$
    \begin{split}\mathbf{A}\mathbf{x}
    = \begin{bmatrix}
    \mathbf{a}^\top_{1} \\
    \mathbf{a}^\top_{2} \\
    \vdots \\
    \mathbf{a}^\top_m \\
    \end{bmatrix}\mathbf{x}
    = \begin{bmatrix}
     \mathbf{a}^\top_{1} \mathbf{x}  \\
     \mathbf{a}^\top_{2} \mathbf{x} \\
    \vdots\\
     \mathbf{a}^\top_{m} \mathbf{x}\\
    \end{bmatrix}.\end{split}
    $$

- 矩阵乘法：`torch.mm(A, B)`

- 范数：一个向量的范数告诉我们一个向量有多大，欧几里得距离是 $L_2$ 范数，`torch.norm(u)`

    $$
    \|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}
    $$
    $L_1$ 范数是向量元素的绝对值之和：`torch.abs(u).sum()`
    $$
    \|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|
    $$
    矩阵的 $L_2$ 范数：弗罗贝尼乌斯范数 `torch.norm(A)`
    $$
    \|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}
    $$

### 微分

- 我们可以将拟合模型的任务分解为两个关键问题：

    （1）*优化*（optimization）：用模型拟合观测数据的过程；

    （2）*泛化*（generalization）：数学原理和实践者的智慧，能够指导我们生成出有效性超出用于训练的数据集本身的模型。

- 梯度：连结一个多元函数对其所有变量的偏导数，可以得到该函数的梯度向量
    $$
    \nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,
    $$

- 常见梯度：假设 $$\mathbf{x}$$ 为 $n$ 维向量，在微分多元函数时经常使用以下规则：

    - 对于所有 $$\mathbf{A} \in \mathbb{R}^{m \times n}$$，都有 $$\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$$
    - 对于所有 $$\mathbf{A} \in \mathbb{R}^{n \times m}$$，都有 $$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} = \mathbf{A}$$
    - 对于所有 $$\mathbf{A} \in \mathbb{R}^{n \times n}$$，都有 $$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x} = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$$
    - $$\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$$

- $f(\mathbf{x})=\|\mathbf{x}\|_2$ 的梯度：
    $$
    \|\mathbf{x}\|_2=\sqrt{\sum_{i=1}^nx_i^2} \\
    设 \ u(x)=\sum_{i=1}^nx_i^2 \\
    D(\|\mathbf{x}\|_2)=D(\sqrt{u(x)})=\frac{1}{2\sqrt{u(x)}}×D(u(x)) \\
    =\frac{1}{2\sqrt{\sum_{i=1}^nx_i^2}}×D(\sum_{i=1}^nx_i^2) \\
    =\frac{1}{2\sqrt{\sum_{i=1}^nx_i^2}}×2\mathbf{x} \\
    =\frac{\mathbf{x}}{\sqrt{\sum_{i=1}^nx_i^2}} \\
    =\frac{\mathbf{x}}{\|\mathbf{x}\|_2}
    $$

### 自动求导

- 深度学习框架通过自动计算导数，即*自动求导*（automatic differentiation），来加快这项工作。实际中，根据我们设计的模型，系统会构建一个*计算图*（computational graph），来跟踪计算是哪些数据通过哪些操作组合起来产生输出。自动求导使系统能够随后反向传播梯度。这里，*反向传播*（backpropagate）只是意味着跟踪整个计算图，填充关于每个参数的偏导数。
- 标量函数关于向量 x 的梯度是向量，并且与 x 具有相同的形状。
- 相关 api
    - 自动求梯度：`x.requires_grad_(True)`
    - 反向传播：`y.backward()`
    - 在默认情况下，PyTorch 会累积梯度，我们需要清除之前的值：`x.grad.zero_()`
    - 对向量的反向传播：`y.sum().backward()` 一个向量是不进行 backward 操作的，而 sum() 后，由于梯度为 1，所以对结果不产生影响。反向传播算法一定要是一个标量才能进行计算。
    - 一个计算图默认情况下不能连续 backward() 多次，Pytorch 的机制是每次调用 .backward() 都会 free 掉所有 buffers，模型中可能有多次 backward()，而前一次 backward() 存储在 buffer 中的梯度，会因为后一次调用 backward() 被 free 掉，可以使用 retain_graph=True 参数

### 概率

- 条件概率：
    $$
    P(B=b|A=a)=\frac{P(A=a,B=b)}{P(A=a)}\in [0,1]
    $$

- 乘法规则：
    $$
    P(A,B)=P(B|A)P(A)=P(A|B)P(B)
    $$

- 贝叶斯定理：
    $$
    P(A|B)=\frac{P(B|A)P(A)}{P(B)}
    $$

- 边际化（求和规则）：
    $$
    P(B)=\sum_AP(A,B)
    $$

- 独立性：两个随机是独立的当且仅当两个随机变量的联合分布是其各自分布的乘积
    $$
    A⊥B \iff P(A,B)=P(A)P(B) \\
    A⊥B|C \iff P(A,B|C)=P(A|C)P(B|C)
    $$

- 假设我们有一系列随机变量，例如 A，B 和 C，其中 B 只依赖于 A，而 C 只依赖于 B，你能简化联合概率 P(A,B,C) 吗？
    $$
    P(A,B,C)=P(C|A,B)P(A,B)=P(C|A,B)P(B|A)P(A)=P(C|B)P(B|A)P(A)
    $$


## 