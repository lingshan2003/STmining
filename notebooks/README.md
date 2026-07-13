# Learning notebooks

Notebooks are intentionally secondary to the tested package. Create exploration notebooks in this order:

1. `01_data_windows.ipynb`: inspect daily periodicity, missingness, and manually verify one window.
2. `02_graph_message_passing.ipynb`: multiply a tiny adjacency matrix by node features by hand.
3. `03_transfer_analysis.ipynb`: load generated CSV results and plot transfer gain.

Each notebook should import code from `crosscity` instead of duplicating preprocessing logic.

