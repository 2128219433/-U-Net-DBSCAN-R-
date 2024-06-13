# 基于U-Net和DBSCAN的R峰检测实验报告

## 引言
心电图（ECG）信号对于诊断各种心脏病至关重要。准确检测R峰对于分析心率变异性和其他心脏参数至关重要。本实验结合了U-Net模型进行初始R峰检测，并使用DBSCAN聚类来优化检测到的峰值。主要目标是提高R峰检测的准确性、灵敏度和阳性预测值。

## 数据准备
### 数据集
本实验使用了MIT-BIH心律失常数据库。

### 预处理步骤
1. **加载ECG信号：** 使用`wfdb`库加载数据集中的ECG信号，主要使用第一个导联信号。
2. **标准化：** 每个ECG信号标准化为零均值和单位方差。具体公式如下：
$
\text{normalized\_signal} = \frac{\text{ecg\_signal} - \mu}{\sigma}
$

   其中，$\mu$ 是信号的均值，$\sigma$ 是信号的标准差。
3. **分割：** 信号分割为5000个样本的片段，以便批处理。
4. **滤波：** 使用三阶Butterworth滤波器应用带通滤波（5-20 Hz）以去除噪声。Butterworth滤波器的传递函数为：
   \[
   H(s) = \frac{1}{\sqrt{1 + \left(\frac{s}{\omega_c}\right)^{2n}}}
   \]
   其中，$n$ 是滤波器的阶数，$\omega_c$ 是截止频率。
5. **QRS检测：** 使用XQRS算法检测QRS波群的位置。

## 模型架构和训练
### U-Net模型
U-Net模型的架构由编码器（下采样路径）和解码器（上采样路径）组成，并具有跳跃连接以保留高分辨率特征。

### 数学表示
- **卷积层：** 卷积操作可以表示为：
  \[
  \text{Conv}(x, w) = x * w
  \]
  其中，$x$ 是输入信号，$w$ 是卷积核，$*$ 表示卷积操作。
- **激活函数：** ReLU激活函数表示为：
  \[
  \text{ReLU}(x) = \max(0, x)
  \]
- **批归一化：** 公式为：
  \[
  \text{BN}(x) = \gamma \left( \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \right) + \beta
  \]
  其中，$\mu$ 和 $\sigma$ 分别是批次的均值和标准差，$\gamma$ 和 $\beta$ 是可学习参数，$\epsilon$ 是一个小常数以避免除零。

### 损失函数
二元交叉熵损失（Binary Cross-Entropy Loss），公式为：
\[
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]
\]
其中，$y_i$ 是真实标签，$p_i$ 是预测概率。

### 优化器
Adam优化器，更新参数的公式为：
\[
\begin{align}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \frac{\alpha \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align}
\]
其中，$g_t$ 是当前梯度，$\alpha$ 是学习率，$\beta_1$ 和 $\beta_2$ 是动量参数。

## 使用DBSCAN的后处理
### 目标
通过对U-Net模型的二值输出进行聚类来优化R峰检测。

### 数学表示
- **二值化：** 使用阈值$T$将U-Net输出二值化：
  \[
  \hat{y}(x) = \begin{cases} 
  1 & \text{if } y(x) > T \\
  0 & \text{otherwise}
  \end{cases}
  \]
- **DBSCAN聚类：** DBSCAN的核心思想是通过密度聚类，将具有足够相邻点的点聚类为一个簇。DBSCAN算法包括以下步骤：
  1. 对于每个点，计算以该点为中心、半径为$\epsilon$的邻域内的点数。
  2. 如果邻域内的点数不小于指定的最小样本数MinPts，则将该点标记为核心点，并将邻域内的所有点包括在同一簇中。
  3. 对每个核心点，将其邻域内的所有点标记为同一簇，重复上述步骤，直到所有点都已被访问。

## 评估指标
使用以下指标评估U-Net和DBSCAN结合方法的性能：
- **灵敏度（召回率）：**
  \[
  \text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  \]
- **阳性预测值（精确度）：**
  \[
  \text{Positive Predictivity} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  \]
- **准确率：**
  \[
  \text{Accuracy} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}
  \]
- **F1评分：**
  \[
  \text{F1} = 2 \cdot \frac{\text{Sensitivity} \cdot \text{Positive Predictivity}}{\text{Sensitivity} + \text{Positive Predictivity}}
  \]
其中，TP（True Positive）是真阳性数，FP（False Positive）是假阳性数，FN（False Negative）是假阴性数。

## 结果
所有测试样本的平均性能指标如下：
- **灵敏度：** 0.9282
- **阳性预测值：** 0.9924
- **准确率：** 0.9276
- **F1评分：** 0.9540

这些结果表明该方法在R峰检测中具有很高的精确度和总体有效性。

## 可视化
下图显示了包含真实R峰和检测到的R峰的ECG信号。绿色点表示真实R峰，红色点表示检测到的R峰：

$$![output](https://github.com/2128219433/-U-Net-DBSCAN-R-/assets/124794397/499159a6-fb91-4129-86d9-14641306d5b8)$$

## 讨论
U-Net和DBSCAN结合在准确检测ECG信号中的R峰方面表现出色。高阳性预测值和F1评分证明了模型的鲁棒性。然而，灵敏度可以进一步提高，这表明一些真实的R峰被漏检。未来的工作可以集中在优化阈值和聚类参数以提高灵敏度。

## 结论
本实验成功应用U-Net模型和DBSCAN聚类检测ECG信号中的R峰，取得了较高的精确度和总体准确性。结果验证了深度学习结合聚类技术在生物医学信号处理中的潜力。
