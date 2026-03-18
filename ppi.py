import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


EDGE_CSV = "ppi_edges_scores.csv"
OUTPUT_PDF = "ppi_network_plot_acta2_mmp12_focus.pdf"
OUTPUT_NODE_CSV = "ppi_focus_acta2_mmp12_nodes.csv"
OUTPUT_EDGE_CSV = "ppi_focus_acta2_mmp12_edges.csv"

CENTERS = ("ACTA2", "MMP12")


def load_graph_from_edge_csv(path: Path) -> nx.Graph:
    graph = nx.Graph()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = (row.get("Node_A") or "").strip()
            v = (row.get("Node_B") or "").strip()
            if not u or not v or u == v:
                continue
            score_raw = (row.get("Score") or "").strip()
            try:
                score = float(score_raw)
            except ValueError:
                score = 0.0
            graph.add_edge(u, v, score=score)
    return graph


def extract_first_degree_star(full_graph: nx.Graph, centers: tuple[str, str]) -> tuple[nx.Graph, dict[str, set[str]]]:
    missing = [node for node in centers if node not in full_graph]
    if missing:
        raise ValueError(f"Missing center node(s) in source graph: {', '.join(missing)}")

    neighbors = {c: set(full_graph.neighbors(c)) for c in centers}
    sub_nodes = set(centers) | neighbors[centers[0]] | neighbors[centers[1]]

    sub_graph = nx.Graph()
    sub_graph.add_nodes_from(sub_nodes)

    # Keep only edges directly connected to ACTA2 or MMP12.
    for c in centers:
        for n in neighbors[c]:
            edge_data = full_graph.get_edge_data(c, n, default={})
            sub_graph.add_edge(c, n, score=float(edge_data.get("score", 0.0)))

    # Keep the center-to-center edge if it exists.
    if full_graph.has_edge(centers[0], centers[1]):
        data = full_graph.get_edge_data(centers[0], centers[1], default={})
        sub_graph.add_edge(centers[0], centers[1], score=float(data.get("score", 0.0)))

    return sub_graph, neighbors


def _arc_positions(nodes: list[str], cx: float, cy: float, radius: float, start_deg: float, end_deg: float) -> dict[str, tuple[float, float]]:
    if not nodes:
        return {}
    if len(nodes) == 1:
        angle = math.radians((start_deg + end_deg) / 2.0)
        return {nodes[0]: (cx + radius * math.cos(angle), cy + radius * math.sin(angle))}

    positions = {}
    for i, node in enumerate(nodes):
        angle_deg = start_deg + (end_deg - start_deg) * i / (len(nodes) - 1)
        angle = math.radians(angle_deg)
        positions[node] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    return positions


def axis_layout(neighbor_map: dict[str, set[str]], centers: tuple[str, str]) -> dict[str, tuple[float, float]]:
    left, right = centers
    left_pos = (-2.1, 0.0)
    right_pos = (2.1, 0.0)

    left_neighbors = neighbor_map[left]
    right_neighbors = neighbor_map[right]

    shared = sorted(left_neighbors & right_neighbors)
    left_only = sorted(left_neighbors - right_neighbors)
    right_only = sorted(right_neighbors - left_neighbors)

    pos: dict[str, tuple[float, float]] = {
        left: left_pos,
        right: right_pos,
    }

    pos.update(_arc_positions(left_only, left_pos[0], left_pos[1], radius=1.95, start_deg=130, end_deg=230))
    pos.update(_arc_positions(right_only, right_pos[0], right_pos[1], radius=1.95, start_deg=-50, end_deg=50))

    if shared:
        span = 2.8
        if len(shared) == 1:
            y_values = [0.0]
        else:
            y_values = [(-span / 2.0) + i * (span / (len(shared) - 1)) for i in range(len(shared))]
        for node, y in zip(shared, y_values):
            pos[node] = (0.0, y)

    return pos


def save_subgraph_tables(graph: nx.Graph, neighbor_map: dict[str, set[str]], centers: tuple[str, str]) -> None:
    left, right = centers
    left_neighbors = neighbor_map[left]
    right_neighbors = neighbor_map[right]

    with Path(OUTPUT_NODE_CSV).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Node", "Group"])
        for node in sorted(graph.nodes()):
            if node == left:
                group = "ACTA2_center"
            elif node == right:
                group = "MMP12_center"
            elif node in left_neighbors and node in right_neighbors:
                group = "shared_neighbor"
            elif node in left_neighbors:
                group = "ACTA2_neighbor"
            else:
                group = "MMP12_neighbor"
            writer.writerow([node, group])

    with Path(OUTPUT_EDGE_CSV).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Node_A", "Node_B", "Score"])
        for u, v, data in sorted(graph.edges(data=True)):
            writer.writerow([u, v, data.get("score", 0.0)])


def draw_focus_subnetwork(graph: nx.Graph, neighbor_map: dict[str, set[str]], centers: tuple[str, str]) -> None:
    left, right = centers
    pos = axis_layout(neighbor_map, centers)

    left_neighbors = neighbor_map[left]
    right_neighbors = neighbor_map[right]

    node_colors = []
    node_sizes = []
    for node in graph.nodes():
        if node == left:
            node_colors.append("#b30000")
            node_sizes.append(3300)
        elif node == right:
            node_colors.append("#e6550d")
            node_sizes.append(3300)
        elif node in left_neighbors and node in right_neighbors:
            node_colors.append("#fdd49e")
            node_sizes.append(1900)
        elif node in left_neighbors:
            node_colors.append("#fcbba1")
            node_sizes.append(1450)
        else:
            node_colors.append("#fdae6b")
            node_sizes.append(1450)

    scores = [float(graph[u][v].get("score", 0.0)) for u, v in graph.edges()]
    if scores:
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            max_s = min_s + 1.0
    else:
        min_s, max_s = 0.0, 1.0

    edge_widths = []
    edge_colors = []
    for u, v in graph.edges():
        score = float(graph[u][v].get("score", 0.0))
        normalized = (score - min_s) / (max_s - min_s)
        edge_widths.append(1.5 + 5.5 * normalized)
        if {u, v} == {left, right}:
            edge_colors.append("#404040")
        else:
            edge_colors.append("#8f8f8f")

    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.75,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="#2f2f2f",
        linewidths=0.8,
        ax=ax,
    )

    center_labels = {left: left, right: right}
    neighbor_labels = {n: n for n in graph.nodes() if n not in centers}
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=center_labels,
        font_size=18,
        font_weight="bold",
        font_color="#111111",
        font_family="Arial",
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=neighbor_labels,
        font_size=18,
        font_color="#1f1f1f",
        font_family="Arial",
        ax=ax,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    edge_path = Path(EDGE_CSV)
    if not edge_path.exists():
        raise FileNotFoundError(f"Input edge CSV not found: {edge_path}")

    full_graph = load_graph_from_edge_csv(edge_path)
    focus_graph, neighbor_map = extract_first_degree_star(full_graph, CENTERS)
    if focus_graph.number_of_edges() == 0:
        raise RuntimeError("Focus subnetwork is empty after filtering.")

    draw_focus_subnetwork(focus_graph, neighbor_map, CENTERS)
    save_subgraph_tables(focus_graph, neighbor_map, CENTERS)

    print(f"Focus PPI figure saved: {OUTPUT_PDF}")
    print(f"Focus node table saved: {OUTPUT_NODE_CSV}")
    print(f"Focus edge table saved: {OUTPUT_EDGE_CSV}")
    print(
        f"Nodes={focus_graph.number_of_nodes()}, Edges={focus_graph.number_of_edges()}, "
        f"ACTA2_neighbors={len(neighbor_map['ACTA2'])}, MMP12_neighbors={len(neighbor_map['MMP12'])}"
    )


if __name__ == "__main__":
    main()
