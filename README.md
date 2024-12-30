# GeoAI-driven spatiotemporal forecasting

## 1. 代码版本介绍及超参数命名

### 1.1 代码版本介绍
* Python Version: 3.8
* Pytorch Version: 1.8.0

### 1.2 超参数命名
* batch_size：批处理的大小
* num_of_nodes：图节点的数量
* num_of_timesteps：时间依赖步长
* in_channels：输入数据的特征数量
* hidden_channels：隐藏层的维度 
* num_of_headers: 多头注意力中“头”的个数
* kernel_sizes: 卷积核的大小
* multi_steps：预测步长

## 2. 基础神经网络算子

### 2.1 Modeling Temporal Dependence(建模时间依赖关系的基础神经网络算子)

* Operator[RNN]v1: 利用PyTorch底层函数自定义线性变换运算，实现**经典循环神经网络算子(RNN算子)**
* Operator[RNN]v2: 利用PyTorch自带的nn.Linear()类完成线性变换运算，实现**经典循环神经网络算子(RNN算子)**
* Operator[RNN]v3: 利用PyTorch自带的F.linear()函数完成线性变换运算，实现**经典循环神经网络算子(RNN算子)**
* Operator[GRU]: 利用PyTorch自带的nn.Linear()类实现**门控循环单元算子(GRU算子)**
* Operator[LSTM]: 利用PyTorch自带的nn.Linear()类实现**长短时记忆网络算子(LSTM算子)**
* Operator[1DCNN]v1: 利用PyTorch自带的nn.Conv1d()类实现**一维卷积神经网络算子(1DCNN算子)**
* Operator[1DCNN]v2: 利用PyTorch自带的F.conv1d()函数**一维卷积神经网络算子(1DCNN算子)**
* Operator[CausalCN]: 利用PyTorch自带的nn.Conv1d()类和padding参数**因果卷积神经网络算子(CausalCN算子)**
* Operator[CausalDCN]v1: 利用PyTorch自带的nn.Conv1d()类、padding和dilation参数实现**因果膨胀卷积神经网络算子(CausalDCN算子)**
* Operator[CausalDCN]v2: 利用PyTorch自带的F.conv1d()类、padding和dilation参数实现**因果膨胀卷积神经网络算子(CausalDCN算子)**
* Operator[TAtten]: 利用PyTorch底层函数实现**时间注意力算子(TAtten算子)**
### 2.2 Modeling Spatial Dependence(建模空间依赖关系的基础神经网络算子)
* Operator[2DCNN]v1: 利用PyTorch自带的nn.Conv2d()类实现**二维卷积神经网络算子(2DCNN算子)**
* Operator[2DCNN]v2: 利用PyTorch自带的F.conv2d()函数实现**二维卷积神经网络算子(2DCNN算子)**
* Operator[2DDCNN]: 利用PyTorch自带的F.conv2d()函数、padding和dilation参数实现**二维膨胀卷积神经网络算子(2DDCNN算子)**
* Operator[SAtten]: 利用PyTorch底层函数实现**空间注意力算子(SAtten算子)**
* Operator[GAT]: 利用PyTorch底层函数实现**图注意力算子(GAT算子)**
* Operator[SpatialGCN]: 利用PyTorch底层函数实现**空间图卷积神经网络算子(SpatialGCN算子)**
### 2.3 Modeling Spatiotemporal Dependence(建模时空依赖关系的基础神经网络算子)
* Operator[ConvGRU]: 利用PyTorch自带的nn.Conv2d()类替换经典GRU算子中的线性变换实现**卷积门控循环单元算子(ConvGRU算子)**
* Operator[ConvLSTM]: 利用PyTorch自带的nn.Conv2d()类替换经典LSTM算子中的线性变换实现**卷积长短时记忆网络算子(ConvLSTM算子)**
* Operator[3DCNN]v1: 利用PyTorch自带的nn.Conv3d()类实现**三维卷积神经网络算子(3DCNN算子)**
* Operator[3DCNN]v2: 利用PyTorch自带的F.conv3d()函数实现**三维卷积神经网络算子(3DCNN算子)**
* Operator[STCNN]: 利用PyTorch自带的nn.Conv3d()类和padding参数实现**时空卷积神经网络算子(STCNN算子)**
* Operator[STDCNN]: 利用PyTorch自带的nn.Conv3d()类、padding和dilation参数实现**时空膨胀卷积神经网络算子(STDCNN算子)**
* Operator[ParallelBlock]: 时间维度采用CausalCN算子、空间维度采用SAtten算子，实现**时空并联块**
* Operator[SerialBlock]: 时间维度采用CausalCN算子、空间维度采用SAtten算子，实现**时空串联块**
