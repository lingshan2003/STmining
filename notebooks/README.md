# Learning notebooks

完整课程结构、先修关系、实验结论与后续计划见 [`../LEARNING_GUIDE.md`](../LEARNING_GUIDE.md)。本目录只作为 notebook 的文件索引；**课程编号始终等于文件名前缀**。

## A. 时间建模基础

| 编号 | Notebook | 主题 |
|---|---|---|
| 01 | `01_data_and_windows.ipynb` | 交通矩阵、滑动窗口与时间切分 |
| 02 | `02_baselines_and_metrics.ipynb` | HA、mask 与预测指标 |
| 03 | `03_shared_mlp_and_training.ipynb` | 共享 MLP 与完整训练循环 |
| 04 | `04_shared_lstm_and_sequence_modeling.ipynb` | LSTM 与序列建模 |

## B. 图消息传递与时空图

| 编号 | Notebook | 主题 |
|---|---|---|
| 05 | `05_graph_and_spatial_message_passing.ipynb` | 邻接矩阵、归一化与空间传播 |
| 06 | `06_stgcn_and_graph_ablation.ipynb` | STGCN 与图结构消融 |
| 07 | `07_graph_direction_diagnostics.ipynb` | 图方向、identity/shuffled 对照与多种子 |

## C. 通用 GNN 任务

| 编号 | Notebook | 主题 |
|---|---|---|
| 08 | `08_static_gnn_node_classification.ipynb` | Cora 节点分类：MLP/GCN/SAGE/GAT |
| 09 | `09_graph_classification_and_transformer.ipynb` | MUTAG 图分类：GIN/GPS 与 global pooling |
| 10 | `10_link_prediction_and_graph_autoencoder.ipynb` | Cora 链路预测：负采样、GAE/VGAE |

## D. 推荐、知识图谱与异构图

| 编号 | Notebook | 主题 |
|---|---|---|
| 11 | `11_recommendation_and_lightgcn.ipynb` | MovieLens 二部图推荐、MF 与 LightGCN |
| 12 | `12_recommendation_graph_diagnostics.ipynb` | 传播深度、噪声、热门度与长尾诊断 |
| 13 | `13_knowledge_graph_and_rgcn.ipynb` | TransE、DistMult、R-GCN 与知识图谱评价 |
| 14 | `14_heterogeneous_product_recommendation.ipynb` | 多行为商品异构图与 HeteroData |
| 15 | `15_hgt_on_dblp.ipynb` | 公开 DBLP 上的 HeteroSAGE 与 HGT |

## E. 动态图

| 编号 | Notebook | 状态 |
|---|---|---|
| 16 | `16_temporal_graph_network.ipynb` | 已完成：JODIE Wikipedia、TGN 与严格时间评价 |

## F. GNN 核心进阶

| 编号 | Notebook | 状态 |
|---|---|---|
| 17 | 待创建 | 下一课：大图 mini-batch、neighbor sampling 与 GraphSAGE |
| 18 | 待创建 | 规划：过平滑、过挤压与异配图 |
| 19 | 待创建 | 规划：图自监督、预训练与 GraphMAE |
| 20 | 待创建 | 规划：图基础模型与图—语言模型 |

Notebook 负责解释与实验编排；可复用的数据、模型和训练逻辑必须从 `crosscity` package 导入，避免在 notebook 中复制出另一套不可测试的实现。
