# Learning notebooks

Notebooks are intentionally secondary to the tested package. The current learning sequence is:

1. `01`–`04`: traffic data, baselines, and temporal models.
2. `05`–`07`: graph message passing, STGCN, and controlled graph ablations.
3. `08_static_gnn_node_classification.ipynb`: MLP, GCN, GraphSAGE, and GAT on Cora.
4. `09_graph_classification_and_transformer.ipynb`: graph mini-batching, GIN, and GPS on MUTAG.

Each notebook should import code from `crosscity` instead of duplicating preprocessing logic.
