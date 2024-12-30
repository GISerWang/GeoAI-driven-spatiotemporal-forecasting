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

* Operator[RNN]v1: 
* Operator[RNN]v2: 
* Operator[RNN]v3: 
* Operator[GRU]: 
* Operator[LSTM]: 
* Operator[1DCNN]v1: 
* Operator[1DCNN]v2: 
* Operator[CausalCN]: 
* Operator[CausalDCN]v1: 
* Operator[CausalDCN]v2: 
* Operator[TAtten]: 
### 2.2 Modeling Spatial Dependence(建模空间依赖关系的基础神经网络算子)
* Operator[2DCNN]v1: 
* Operator[2DCNN]v2:
* Operator[2DDCNN]:
* Operator[SAtten]: 
* Operator[GAT]: 
* Operator[SpatialGCN]: 
### 2.3 Modeling Spatiotemporal Dependence(建模时空依赖关系的基础神经网络算子)
* Operator[ConvGRU]: 
* Operator[ConvLSTM]:
* Operator[3DCNN]v1:
* Operator[3DCNN]v2:
* Operator[STCNN]:
* Operator[STDCNN]:
