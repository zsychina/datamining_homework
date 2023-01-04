# $$数\ 据\ 挖\ 掘\ 大\ 作\ 业$$

$$朱\ 思\ 源\ \ \ \ 20202261013\ \ \ \ 电 \ 自\ 2001$$

$$siyuan100@mail.dlut.edu.cn$$

\* 说明：这份作业我是用markdown写的，但是网上的markdown转pdf服务似乎多少都有点问题，导致含有中文的数学公式在导出的时候效果不太好。所以我会把pdf文件和markdown文件一起附在附件上。还请老师谅解。

## 1. 什么是数据挖掘？数据挖掘的主要步骤是什么？（C1）

### **(1) 什么是数据挖掘**

数据挖掘是从海量数据中探索出有用的信息与知识。  

数据挖掘涉及使用机器学习和统计分析等先进技术从大型数据集中提取有用的见解和知识。它是一个多学科领域，结合了计算机科学、统计学和领域专业知识来分析数据中的模式和关系。  

有许多不同的技术可用于数据挖掘，包括决策树、聚类和神经网络。这些技术可以应用于各种不同类型的数据，包括数据库中的结构化数据和文本、图像和视频等非结构化数据。  

数据挖掘的应用范围很广，包括欺诈检测、市场分析和科学发现。它是各种规模的企业和组织做出明智决策、发现新机会和解决复杂问题的重要工具。  

### **(2) 数据挖掘的主要步骤是什么**

- 数据清洗：去除噪声与不一致的数据  

- 数据集成：多源数据的合并  

- 数据筛选：对需要进行分析的数据进行检索  

- 数据变换：对数据进行变换，以得到适当的格式为后续的数据总结与聚合进行服务  

- 数据挖掘：利用智能方法对数据模式进行抽取  

- 模式评价：对抽取的模式进行评价，并选择最感兴趣的模式  

- 知识展示：对数据挖掘获取的知识进行可视化等展示  

## 2. 试阐述本课程所讨论过的几种数据属性。如何计算各种数据属性的相异性？数据的统计性描述指标都有哪些，请详细阐述？（C2）

### **(1) 数据属性**

- 标称属性：用于直接描述对象的某种属性  

- 二值属性：二值属性是标称属性的特殊形式，只分为0和1  

- 次序属性：次序属性是指数据对应于具有一定意义的顺序或者等级  

- 数值属性：数值属性是定量的属性，即可测的品质，分为等距尺度与等比尺度  

### **(2) 如何计算各种数据属性的相异性**

核心在于计算相异性矩阵

相异性矩阵 =

$$
\begin{bmatrix}
0\\
d(2,1)&0\\
d(3,1)&d(3,2)&0\\
\vdots&\vdots&\vdots&\ddots\\
d(n,1)&d(n,2)&\cdots&\cdots&0\\
\end{bmatrix}
$$

$d(i,j)$ 称为相异性度量

其中，

- 标称数据的相异性度量为 $$d(i,j)=\frac{p-m}{p}$$

- 二值数据的相异性度量为 $$d(i,j)=\frac{r+s}{q+r+s+t}$$

  将 $i=j=0$ 的个数 $t$ 忽略，得
  
  $$d(i,j)=\frac{r+s}{q+r+s}$$

- 对于数值属性数据，$d(i,j)$ 可以由以下公式给出：

  - 闵可夫斯基距离：

    $$d(i,j)=(|x_{i1}-x_{j1}|^h+|x_{i2}-x_{j2}|^h+\cdots+|x_{ip}-x_{jp}|^h)^{\frac{1}{h}}=||x_i-x_j||_h$$

  - 欧几里得距离：令 $h=2$ 时，有

    $$d(i,j)=(|x_{i1}-x_{j1}|^2+|x_{i2}-x_{j2}|^2+\cdots+|x_{ip}-x_{jp}|^2)^{\frac{1}{2}}=||x_i-x_j||_2$$

  - 曼哈顿距离：令 $h=1$ 时，有

    $$d(i,j)=|x_{i1}-x_{j1}|+|x_{i2}-x_{j2}|+\cdots+|x_{ip}-x_{jp}|=||x_i-x_j||$$

  - 切比雪夫距离：令 $h=\infty$ 时，有

    $$d(i,j)=||x_i-x_j||_\infty=\max\{|x_{i1}-x_{j1}|,|x_{i2}-x_{j2}|,\cdots,|x_{ip}-x_{jp}|\}$$

- 对次序数据，相异性计算则先要进行 $z_{if}=\frac{r_{if}-1}{M_f-1}$ 变换，进行数值属性的归一化，再用基于数值属性的闵可夫斯基距离进行计算

- 对混合属性，则要先归一化到 $[0,1]$ 的区间内，然后考虑

  $$d(i,j)=\frac{\sum_{f=1}^{p}\delta_{ij}^{(f)}d_{ij}^{(f)}}{\sum_{f=1}^{p}\delta_{ij}^{(f)}}$$

  其中 $\delta_{ij}^{(f)}$ 为指示函数：

  $$\delta_{ij}^{(f)}=\begin{cases}
  0,\ \ (1)\ x_{if}\ 或\ x_{jf}\ 缺\ 失；(2)\ f\ 为\ 非\ 对\ 称\ 二\ 值\ 属\ 性\ 且\ x_{if}=x_{if}=0\\
  1,\ \ 其\ 他\\
  \end{cases}$$

### **(3) 阐述数据的统计性描述指标**

- 中央趋势：

  $$\overline{x}_w=\frac{\sum_{l=1}^{N}w_ix_i}{\sum_{l=1}^{N}w_i}$$

- 中位数：一组次序数据集位于中间的值

  $$中\ 位\ 数=L_1+(\frac{N/2-\sum_l{频\ 度}_l}{{频\ 度}_{median}})\cdot 区\ 间\ 宽\ 度$$

- 众数：表示一组数据中出现频率最高的数值，有**经验公式**
  
  $$平\ 均\ 数-众\ 数\approx 3\times(\ 平\ 均\ 数-中 \ 位\ 数\ )$$

- 分散程度

  - 值域：最大值与最小值的差

  - 四分位数：表示有 25% 的数据比这个分位点小（或大）

  - 四分间距：第一个与第三个四分位点之间的差成为四分间距

  - 方差

    $$\sigma^2=\frac{1}{N}\sum_{l=1}^N(x_i-\overline{x})^2$$

  - 标准差：方差的平方根 $\sigma$ 即为标准差

- 离群点：样本落入 $Q_1-1.5\times IQR$ 以下或 $Q_3+1.5\times IQR$ 以上区域时称为离群值（离群点）

- 五数概括法：为了兼顾分布的端点，使用样本的**最小值、$Q_1$、中位数、$Q_3$、最大值**来综合描述数据的分布情况。五点概括法的可视化结果为箱形图

## 3. 形如 $\boldsymbol{y}_i=\boldsymbol{x}_i^T\boldsymbol{\beta}+\boldsymbol{\varepsilon}_i$ 的线性回归模型的适用场景是什么？假设数据集为 $\{(y_i, x_i)|i=1,2,\cdots,n\}$ 时，分别推导该模型的最小二乘与最大似然估计结果？(C3)

### **(1) 适用场景**

形如 $\boldsymbol{y}_i=\boldsymbol{x}_i^T\boldsymbol{\beta}+\boldsymbol{\varepsilon}_i$ 的是多自变量线性回归模型。多自变量线性回归模型用于根据多个自变量的值来预测因变量的值。 它用于分析多个自变量与单个因变量之间的关系。

适用场景举例：

- 根据房屋的大小、卧室数量、浴室数量、位置和其他特征预测房屋的价格

- 根据预算、目标受众和促销类型预测营销活动的成功率

- 根据客户的年龄、收入和其他人口统计数据预测客户进行购买的可能性

### **(2) 推导估计结果**

- 最小二乘法

  误差向量表示为

  $$\boldsymbol{\varepsilon}=\boldsymbol{y}-\boldsymbol{X\beta}$$

  目标函数可写为关于参数向量 $\beta$ 的函数 $S(\boldsymbol{\beta})$

  $$S(\boldsymbol{\beta})=\sum_{l=1}^{n}\varepsilon_i^2=\boldsymbol{\varepsilon}^T\boldsymbol{\varepsilon}=(\boldsymbol{y}-\boldsymbol{X\beta})^T(\boldsymbol{y}-\boldsymbol{X\beta})$$

  可以理解为线性回归模型在各个数据点的误差项的平方和

  $$\boldsymbol{\hat{\beta}}_{LS}=\underset{\beta}{\arg\min}(\boldsymbol{y}-\boldsymbol{X\beta})^T(\boldsymbol{y}-\boldsymbol{X\beta})$$

  $S(\boldsymbol{\beta})$ 是关于 $\beta$ 的凸函数，如果最优化问题的解存在，则该解为全局最优解。

  由于

  $$S(\boldsymbol{\beta})=\boldsymbol{y}^T\boldsymbol{y}-\boldsymbol{y}^T\boldsymbol{X\beta}-\boldsymbol{\beta}^T\boldsymbol{X}^T\boldsymbol{y}+\boldsymbol{\beta}^T\boldsymbol{X}^T\boldsymbol{X\beta}=-2\boldsymbol{y}^T\boldsymbol{x\beta}+\boldsymbol{\beta}^T\boldsymbol{X}^T\boldsymbol{X\beta}$$

  解如下方程

  $$\frac{\partial S(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}}=-2\boldsymbol{X}^T\boldsymbol{y}+2\boldsymbol{X}^T\boldsymbol{X\beta}=0$$

  如果 $(\boldsymbol{X}^T\boldsymbol{X})^{-1}$ 存在，则

  $$\boldsymbol{\hat{\beta}}_{LS}=(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}$$

  此时，回归模型在各个数据点 $y_i$ 的估计值为

  $$\hat{y}_i=\boldsymbol{x}_i^T\boldsymbol{\hat{\beta}}_{LS}$$

- 最大似然估计

  似然函数定义如下

  $$L(\theta|y_1,\cdots,y_n)=f(y_1,\cdots,y_n|\theta)$$

  最大似然法则是在参数空间 $\Theta$ 中产生给定数据 $D$ 概论最大的参数，即

  $$\hat{\theta}_{MLE}=\underset{\theta\in\Theta}{\arg\max}L(\theta|y_1,y_2,\cdots,y_n)$$

  误差项 $\boldsymbol{\varepsilon}$ 的密度函数为

  $$f(\boldsymbol{\varepsilon})=\frac{1}{(2\pi\sigma^2)^{n/2}}\exp(-\frac{1}{2\sigma^2}\boldsymbol{\varepsilon}^T\boldsymbol{\varepsilon})=\prod_{i=1}^n\frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}\varepsilon_i^2)$$

  由于 $\boldsymbol{y}_i=\boldsymbol{x}_i^T\boldsymbol{\beta}+\boldsymbol{\varepsilon}_i$

  $$L(\boldsymbol{\beta},\sigma^2)=f(\boldsymbol{y}|\boldsymbol{X};\boldsymbol{\beta},\sigma^2)=\frac{1}{(2\pi\sigma^2)^{n/2}}\exp\{-\frac{1}{2\sigma^2}(\boldsymbol{y}-\boldsymbol{X\beta})^T(\boldsymbol{y}-\boldsymbol{X\beta})\}$$

  $L(\boldsymbol{\beta},\sigma^2)$ 是关于 $\boldsymbol{\beta},\sigma^2$ 的凹函数，如果最大化的最优解存在，则其解是全局最优的

  对似然函数取对数，得到对数似然函数

  $$l(\boldsymbol{\beta},\sigma^2)=\log L(\boldsymbol{\beta},\sigma^2)=-\frac{n}{2}\log (2\pi\sigma^2)-\frac{1}{2\sigma^2}(\boldsymbol{y}-\boldsymbol{X\beta})^T(\boldsymbol{y}-\boldsymbol{X\beta})$$

  对 $l(\boldsymbol{\beta},\sigma^2)$ 求关于 $\boldsymbol{\beta},\sigma^2$ 的偏导数等于零时的解，即

  $$
  \begin{cases}
    \frac{\partial l(\boldsymbol{\beta},\sigma^2)}{\partial \boldsymbol{\beta}}=\frac{1}{\sigma^2}(\boldsymbol{X}^T\boldsymbol{y}+2\boldsymbol{X}^T\boldsymbol{X\beta})=0\\
    \frac{\partial l(\boldsymbol{\beta},\sigma^2)}{\partial \boldsymbol{\sigma^2}}=-\frac{1}{2\sigma^2}+\frac{1}{2\sigma^4}(\boldsymbol{y}-\boldsymbol{X\beta})^T(\boldsymbol{y}-\boldsymbol{X\beta})=0
  \end{cases}
  $$

  可得到最大似然解

  $$\boldsymbol{\hat{\beta}}_{MLE}=(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}$$

  $$\hat{\sigma}_{MLE}^2=\frac{1}{n}(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\hat{\beta}}_{MLE})^T(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\hat{\beta}}_{MLE})$$

## 4. 阐述分类模型：逻辑回归、Fisher判别分析、⽀持向量机的基本原理，并分析三种⽅法的不同。（C3, C4, C5）

### **(1) 逻辑回归基本原理**

逻辑回归的基本原理是用逻辑函数把线性回归的结果 $(-\infty,+\infty)$ 映射到 $(0,1)$。

逻辑函数为

$$f(x)=\frac{1}{1+e^{-x}}$$

需要用最大似然法进行逻辑回归模型的参数估计。逻辑回归模型对二值属性的随机变量发生概率 $Pr(Y=1|x)$ 以及 $Pr(Y=0|x)=1-Pr(Y=1|x)$ 进行建模，需要用到伯努利分布，即

$$f(y|\pi)=\pi^y(1-\pi)^{1-y},\ y=0,1$$

假设有 $n$ 组数据

$$\{(x_i,y_i)|i=1,2,\cdots,n,y_i\in\{0,1\},x_i\in\mathbb{R}^p\}$$

逻辑回归模型的似然函数为

$$L(\boldsymbol{\beta})=\prod_{i=1}^n\frac{\exp (y_i\boldsymbol{\beta}^T\boldsymbol{x}_i)}{1+\exp(\boldsymbol{\beta}^T\boldsymbol{x}_i)}$$

对数似然函数为

$$l(\boldsymbol{\beta})=\log L(\boldsymbol{\beta})=\sum_{i=1}^ny_i\boldsymbol{\beta}^Tx_i-\sum_{i=1}^n\log\{1+\exp (\boldsymbol{\beta}^Tx_i)\}$$

由于无法求解 $\frac {\partial l(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}}=0$，故逐次使用 Fisher-scoring 法进行求解

对 $l(\boldsymbol{\beta})$ 求关于 $\boldsymbol{\beta}$ 的二次偏导数

$$\frac{\partial^2l(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}\partial \boldsymbol{\beta}^T}=-X^T\Pi(l_n-\Pi)X$$

1. 对 $\boldsymbol{\beta}$ 取初始值 $\boldsymbol{\beta}^{(0)}$

2. 迭代到 $r$ 步得到 $\boldsymbol{\beta}^{(r)}$，在第 $r+1$ 步做如下更新

    $$\boldsymbol{\beta}^{(r+1)}=[X^T\Pi^{(r)}(l_n-\Pi^{(r)})]^{-1}X^T\Pi^{(r)}(l_n-\Pi^{(r)})\cdot[X\Pi^{(r)}+\{\Pi^{(r)}(l_n-\Pi^{(r)})\}^{-1}(\boldsymbol{y}-\Pi^{(r)}1)]$$

    其中，$\Pi^{(r)}$ 通过 $\boldsymbol{\beta}$ 计算得到

3. 判断收敛，如果不收敛，重复迭代

### **(2) Fisher判别分析基本原理**

假设有 $n_1+n_2$ 组数据：

$$G_1:x_1^{(1)},x_2^{(1)},\cdots,x_{n_1}^{(1)}$$

$$G_2:x_1^{(2)},x_2^{(2)},\cdots,x_{n_2}^{(2)}$$

其中，$x_i^{(j)}\in\mathbb{R}^p$，Fisher 判别的基本思想就是寻找最优系数向量 $w_{OPT}\in\mathbb{R}^p$，使得两组投影

$$G_1:\{ y_i^{(1)}=w_{OPT}^Tx_i^{(1)} | i=1,2,\cdots,n_1 \}$$

$$G_2:\{ y_i^{(2)}=w_{OPT}^Tx_i^{(2)} | i=1,2,\cdots,n_2 \}$$

定义群的投影平均如下

$$\mu_{y^{(i)}}=\frac{1}{n_j}\sum_{i=1}^{n_j}y_i^{(i)}=\frac{1}{n_j}\sum_{i=1}^{n_j}w_{OPT}^Tx_i^{(j)},j=1,2$$

两个群的投影平均距离最小，即对于任意的系数向量 $w$ 产生的投影，有

$$(\mu_{y^{(1)}}-\mu_{y^{(2)}})^2<(\mu_{z^{(1)}}-\mu_{z^{(2)}})^2$$

这个平均距离称为在 $y$ 轴上的**群间方差**

$G_1:\{ y_i^{(1)}=w_{OPT}^Tx_i^{(1)} | i=1,2,\cdots,n_1 \}\ G_2:\{ y_i^{(2)}=w_{OPT}^Tx_i^{(2)} | i=1,2,\cdots,n_2 \}$ 的样本方差要达到最小，即在两群的在 $y$ 轴上的投影不确定性最小，两个群在 $y$ 轴上的投影样本方差可以按照以下方式综合计算

$$群\ 内\ 方\ 差=\frac{1}{n_1+n_2-2}[(n_1-1)\times S_{G_1}+(n_2-1)\times S_{G_2}]$$

$S_{G_j}$ 为 $j$ 群在 $y$ 轴上投影的样本方差，成为**群内方差**

为探索最优系数向量 $w_{OPT}\in\mathbb{R}^p$，使群间方差最大且群内方差最小，定义以下目标

$$w_{OPT}=\underset{w}{\arg\max}\lambda(w),\lambda(w)=\frac{群\ 间\ 方\ 差}{群\ 内\ 方 \ 差}$$

计算两群的样本均值和样本方差，得

$$G_j:\overline{x}_j=\frac{1}{n_j}\sum_{i=1}^{n_j}x_i^{(j)},j=1,2$$

$$S_j=\frac{1}{n_j-1}\sum_{i=1}^{n_j}(x_i^{(j)}-\overline{x}_j)(x_i^{(j)}-\overline{x}_j)^T,j=1,2$$

由 $y=w^Tx$ 得到投影后得样本平均为

$$\mu_{y_j}=\frac{1}{n_j}\sum_{i=1}^{n_j}w^Tx_i^{(j)}=w^T\overline{x}_j,j=1,2$$

样本方差为

$$S_{y_j}=\frac{1}{n_j-1}\sum_{i=1}^{n_j}(y_i^{(j)}-\mu_{y_j})^2=w^TS_jw$$

因此，Fisher 判别目标函数可以写为

$$\lambda(w)=\frac{\{w^T(\overline{x}_1-\overline{x}_2)^2\}}{w^TS_ww}=\frac{w^TS_Bw}{w^TS_ww}$$

其中，$S_B=(\overline{x}_1-\overline{x}_2)(\overline{x}_1-\overline{x}_2)^T,S_2=(n_1+n_2-2)^{-1}((n_1-1)S_1+(n_2-1)S_2)$

### **(3) 支持向量机基本原理**

SVM 是一种可用于分类或回归任务的监督学习算法。SVM 的基本原理是在高维空间中找到最大程度地分离两个类的超平面。超平面与任一类中最近的数据点之间的距离称为边距。目标是选择一个超平面，**该超平面与训练集中的任何点之间的边距最大**，从而使新数据被正确分类的机会更大。

决策边界是通过找到最大程度地分离两个类的超平面来构建的，即

$$w^Tx + b = 0$$

其中，$w$ 是超平面的法向量，$b$ 是偏置项。

超平面一侧的点被分配到一个类别，而另一侧的点被分配到另一个类别。

超平面和点 $x_i$ 之间的距离由下式给出：

$$d = \frac{|w^T x_i + b|}{||w||}$$

每个类中最近点与超平面之间的距离称为边距。目标是选择最大分离两个类的超平面，这意味着找到具有最大边距的超平面。

为找到最大间隔超平面而求解的优化问题可以写为：

$$\underset{w}{\arg\max}M$$

即

$$y_i(w^T x_i + b) \ge M,i = 1, 2, ..., n$$

其中 $M$ 是边距，$y_i$ 是 $x_i$ 的标签。

### **(4) 不同**

- 逻辑回归

  逻辑回归是二元分类的线性模型。它估计一个实例属于特定类别的概率，然后预测概率最高的类别。 逻辑回归也可以通过使用一对多的方法扩展到多类分类。

- Fisher 判别

  Fisher 判别式是一种线性分类器，它可以找到最能区分两个类的特征的线性组合。 它通过最大化类间方差与类内方差的比率来实现这一点。

- 支持向量机

  SVM 是一种线性分类器，它试图找到最大程度地分离两个类的超平面。 除了寻找超平面之外，SVM 还具有使两个类的最近实例之间的距离最大化的边距。 这有助于提高模型的泛化能力。 SVM 也可以通过核技巧扩展到非线性分类。

总的来说，逻辑回归和支持向量机在实践中的应用更为广泛，因为它们鲁棒性更强，并且一般有更好的性能。Fisher 判别在实践中不太常用。
