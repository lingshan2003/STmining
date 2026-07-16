# Learning notebooks

Notebooks are intentionally secondary to the tested package. The current learning sequence is:

1. `01`–`04`: traffic data, baselines, and temporal models.
2. `05`–`07`: graph message passing, STGCN, and controlled graph ablations.
3. `08_static_gnn_node_classification.ipynb`: MLP, GCN, GraphSAGE, and GAT on Cora.
4. `09_graph_classification_and_transformer.ipynb`: graph mini-batching, GIN, and GPS on MUTAG.
5. `10_link_prediction_and_graph_autoencoder.ipynb`: negative sampling, GAE, and VGAE on Cora.
6. `11_recommendation_and_lightgcn.ipynb`: bipartite recommendation, BPR, and LightGCN.
7. `12_recommendation_graph_diagnostics.ipynb`: graph density, propagation depth,
   layer weights, seed variance, popularity bias, and long-tail recommendation.
8. `13_knowledge_graph_and_rgcn.ipynb`: triples, relation embeddings, TransE,
   DistMult, R-GCN, corrupted negatives, filtered MRR, and Hits@K.
9. `14_heterogeneous_product_recommendation.ipynb`: typed nodes, multi-behaviour
   product graphs, PyG HeteroData, relation-specific GraphSAGE, and purchase ranking.
10. `15_hgt_on_dblp.ipynb`: public DBLP, type-specific feature projection,
    heterogeneous attention, multi-head HGT, accuracy, and macro-F1.

Each notebook should import code from `crosscity` instead of duplicating preprocessing logic.
