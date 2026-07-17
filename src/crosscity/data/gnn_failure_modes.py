from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import coalesce, to_undirected

from crosscity.data.citation import NodeClassificationData


def load_wikipedia_network(
    root: str | Path, name: str = "chameleon", split: int = 0
) -> NodeClassificationData:
    """Load one official Geom-GCN split of Chameleon or Squirrel."""
    key = name.lower()
    if key not in {"chameleon", "squirrel"}:
        raise ValueError(f"unsupported WikipediaNetwork dataset: {name}")
    dataset = WikipediaNetwork(
        root=str(Path(root)),
        name=key,
        geom_gcn_preprocess=True,
        transform=NormalizeFeatures(),
    )
    graph = dataset[0]
    if not 0 <= split < graph.train_mask.shape[1]:
        raise ValueError(f"split must be in [0, {graph.train_mask.shape[1] - 1}]")
    return NodeClassificationData(
        x=graph.x,
        y=graph.y,
        edge_index=to_undirected(graph.edge_index),
        train_mask=graph.train_mask[:, split],
        val_mask=graph.val_mask[:, split],
        test_mask=graph.test_mask[:, split],
    )


def edge_homophily(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of directed edge entries whose endpoints have the same label."""
    if edge_index.numel() == 0:
        return float("nan")
    source, target = edge_index
    return float((labels[source] == labels[target]).float().mean())


def feature_knn_edges(x: torch.Tensor, k: int = 10, chunk_size: int = 1024) -> torch.Tensor:
    """Build a label-blind cosine k-NN graph without materializing an N x N matrix."""
    if not 0 < k < x.shape[0]:
        raise ValueError("k must be positive and smaller than the number of nodes")
    normalized = torch.nn.functional.normalize(x.float(), p=2, dim=-1)
    sources, targets = [], []
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        similarities = normalized[start:stop] @ normalized.T
        rows = torch.arange(stop - start, device=x.device)
        similarities[rows, torch.arange(start, stop, device=x.device)] = -torch.inf
        neighbors = similarities.topk(k, dim=1).indices
        targets.append(torch.arange(start, stop, device=x.device).repeat_interleave(k))
        sources.append(neighbors.reshape(-1))
    edge_index = torch.stack([torch.cat(sources), torch.cat(targets)])
    return coalesce(to_undirected(edge_index), num_nodes=x.shape[0])


def rewire_by_features(data: NodeClassificationData, k: int = 10) -> NodeClassificationData:
    """Replace the observed graph by a cosine feature k-NN graph; labels remain unused."""
    return replace(data, edge_index=feature_knn_edges(data.x, k=k))


@dataclass(frozen=True)
class TreeRetrievalBenchmark:
    """Disconnected binary trees with supervision only at each root node."""

    data: NodeClassificationData
    roots: torch.Tensor
    leaves: torch.Tensor
    tree_ids: torch.Tensor
    depth: int

    def with_root_shortcuts(self) -> NodeClassificationData:
        """Add bidirectional leaf-root edges while preserving all features and splits."""
        root_per_leaf = self.roots[self.tree_ids[self.leaves]]
        shortcuts = torch.stack(
            [torch.cat([self.leaves, root_per_leaf]), torch.cat([root_per_leaf, self.leaves])]
        )
        edges = coalesce(
            torch.cat([self.data.edge_index, shortcuts], dim=1),
            num_nodes=self.data.num_nodes,
        )
        return replace(self.data, edge_index=edges)


def make_tree_retrieval_benchmark(
    num_trees: int = 300,
    depth: int = 5,
    seed: int = 42,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> TreeRetrievalBenchmark:
    """Create a long-range task: each root must recover one marked leaf's random bit.

    Every tree is disconnected from all others.  A leaf has two features: its random
    bit and a marker indicating whether it is the queried leaf.  Only roots are
    supervised, so the answer must travel ``depth`` edges in the original graph.
    """
    if num_trees < 5 or depth < 1:
        raise ValueError("num_trees must be at least 5 and depth must be positive")
    nodes_per_tree = 2 ** (depth + 1) - 1
    first_leaf = 2**depth - 1
    leaves_per_tree = 2**depth
    generator = torch.Generator().manual_seed(seed)

    total_nodes = num_trees * nodes_per_tree
    x = torch.zeros((total_nodes, 3), dtype=torch.float32)
    y = torch.zeros(total_nodes, dtype=torch.long)
    tree_ids = torch.arange(num_trees).repeat_interleave(nodes_per_tree)
    roots = torch.arange(num_trees) * nodes_per_tree
    all_leaves = []
    edge_parts = []

    base_children = torch.arange(1, nodes_per_tree)
    base_parents = (base_children - 1) // 2
    for tree in range(num_trees):
        offset = tree * nodes_per_tree
        leaves = offset + torch.arange(first_leaf, nodes_per_tree)
        all_leaves.append(leaves)
        bits = torch.randint(0, 2, (leaves_per_tree,), generator=generator)
        query = int(torch.randint(leaves_per_tree, (1,), generator=generator))
        x[leaves, 0] = bits.float()
        x[leaves[query], 1] = 1.0
        x[offset : offset + nodes_per_tree, 2] = 1.0
        y[offset] = bits[query]
        parents, children = offset + base_parents, offset + base_children
        edge_parts.append(
            torch.stack([torch.cat([parents, children]), torch.cat([children, parents])])
        )

    order = torch.randperm(num_trees, generator=generator)
    train_end = int(train_fraction * num_trees)
    val_end = train_end + int(val_fraction * num_trees)
    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    train_mask[roots[order[:train_end]]] = True
    val_mask[roots[order[train_end:val_end]]] = True
    test_mask[roots[order[val_end:]]] = True
    data = NodeClassificationData(
        x=x,
        y=y,
        edge_index=torch.cat(edge_parts, dim=1),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return TreeRetrievalBenchmark(
        data=data,
        roots=roots,
        leaves=torch.cat(all_leaves),
        tree_ids=tree_ids,
        depth=depth,
    )
