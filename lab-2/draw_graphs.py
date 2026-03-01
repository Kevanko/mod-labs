"""
Построение визуализаций графов для отчёта (лаба 2, CPM).
Сохраняет изображения в папку images/.
"""
import os
import sys
import importlib.util
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Подгрузка lab-2.py (имя с дефисом — не валидный идентификатор модуля)
spec = importlib.util.spec_from_file_location("lab2", os.path.join(SCRIPT_DIR, "lab-2.py"))
lab2 = importlib.util.module_from_spec(spec)
sys.modules["lab2"] = lab2
spec.loader.exec_module(lab2)
GRAPH_VARIANT_17 = lab2.GRAPH_VARIANT_17
GRAPH_CUSTOM = lab2.GRAPH_CUSTOM
run_cpm = lab2.run_cpm

IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Единый стиль для всех графов (как у варианта 17)
STYLE = {
    "node_color": "lightblue",
    "node_size": 1200,
    "edgecolors": "steelblue",
    "linewidths": 2,
    "edge_width": 2.5,
    "edge_color": "steelblue",
    "font_size": 14,
    "font_color": "darkblue",
    "edge_label_font_size": 11,
    "edge_label_bbox": dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="steelblue", linewidth=1.5),
    "figsize": (9, 8),
    # Стрелки направления: крупнее и с отступом от узлов, чтобы хорошо были видны
    "arrowsize": 40,
    "arrowstyle": "-|>",
    "arrow_margin": 15,
}
# Начальные вершины (работы без предшественника) — другой цвет и рамка
START_NODE_STYLE = {
    "node_color": "palegreen",
    "edgecolors": "darkgreen",
    "linewidths": 3,
}
# Завершающие вершины (работы, после которых нет других) — другой цвет и рамка
END_NODE_STYLE = {
    "node_color": "moccasin",
    "edgecolors": "darkorange",
    "linewidths": 3,
}

# Схема варианта 17: узлы A,B,C,D,E,F, цикл с весами на рёбрах
VARIANT17_SCHEME_EDGES = [
    ("A", "B", 5), ("A", "C", 3), ("B", "D", 6), ("C", "E", 4), ("D", "F", 5), ("E", "F", 5),
]
VARIANT17_CRITICAL_EDGES = {("A", "B"), ("B", "D"), ("D", "F")}
VARIANT17_POS = {"A": (0, 0), "B": (1, 0.5), "C": (1, -0.5), "D": (2, 0.5), "E": (2, -0.5), "F": (3, 0)}

# Собственный граф с одним истоком: события 1=A..8=H (без 4); рёбра по GRAPH_CUSTOM
CUSTOM_SCHEME_EDGES = [
    ("A", "B", 4), ("A", "C", 1), ("B", "E", 4), ("C", "E", 1), ("C", "F", 5), ("E", "F", 2), ("F", "G", 3), ("G", "H", 4),
]
CUSTOM_POS = {"A": (0, 0), "B": (1, 0.6), "C": (1, -0.6), "E": (2, 0.3), "F": (2.5, -0.3), "G": (3, 0), "H": (4, 0)}


def draw_variant17_scheme():
    """Схема варианта 17: A–B, A–C, B–D, C–E, D–F, E–F; критический путь A→B→D→F выделен красным; граф ориентированный."""
    G = nx.DiGraph()
    for u, v, w in VARIANT17_SCHEME_EDGES:
        G.add_edge(u, v, weight=w)

    pos = VARIANT17_POS
    edge_labels = {(u, v): str(w) for u, v, w in VARIANT17_SCHEME_EDGES}

    crit_edges = [(u, v) for u, v, w in VARIANT17_SCHEME_EDGES if (u, v) in VARIANT17_CRITICAL_EDGES]
    other_edges = [(u, v) for u, v, w in VARIANT17_SCHEME_EDGES if (u, v) not in VARIANT17_CRITICAL_EDGES]
    start_nodes_v17 = ["A"]   # единственная начальная работа (1→2)
    end_nodes_v17 = ["F"]    # последняя работа (5→6, завершение проекта)

    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    other_nodes_v17 = [n for n in G.nodes() if n not in start_nodes_v17 and n not in end_nodes_v17]
    nx.draw_networkx_nodes(
        G, pos, nodelist=other_nodes_v17, node_color=STYLE["node_color"], node_size=STYLE["node_size"],
        edgecolors=STYLE["edgecolors"], linewidths=STYLE["linewidths"], ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=start_nodes_v17, node_size=STYLE["node_size"],
        **START_NODE_STYLE, ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=end_nodes_v17, node_size=STYLE["node_size"],
        **END_NODE_STYLE, ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=STYLE["font_size"], font_color=STYLE["font_color"], ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color=STYLE["edge_color"], width=STYLE["edge_width"], ax=ax,
        arrows=True, arrowsize=STYLE["arrowsize"], arrowstyle=STYLE["arrowstyle"],
        min_source_margin=STYLE["arrow_margin"], min_target_margin=STYLE["arrow_margin"])
    nx.draw_networkx_edges(G, pos, edgelist=crit_edges, edge_color="crimson", width=STYLE["edge_width"], ax=ax,
        arrows=True, arrowsize=STYLE["arrowsize"], arrowstyle=STYLE["arrowstyle"],
        min_source_margin=STYLE["arrow_margin"], min_target_margin=STYLE["arrow_margin"])
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=STYLE["edge_label_font_size"], bbox=STYLE["edge_label_bbox"], ax=ax)

    ax.set_title("Граф по варианту 17 (ориентированный). Начало (A) — зелёная рамка, окончание (F) — оранжевая. Критический путь A→B→D→F — красный.", fontsize=STYLE["font_size"])
    ax.axis("off")
    plt.tight_layout()
    base = os.path.join(IMAGES_DIR, "graph_variant17")
    plt.savefig(base + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Сохранено:", base + ".svg", "и", base + ".png")


def draw_custom_scheme():
    """Собственный граф: узлы A–H, на рёбрах веса; критический путь из CPM выделен красным."""
    res = run_cpm(GRAPH_CUSTOM)
    # Критический путь по узлам: 1,2,5,6,7,8 → A,B,E,F,G,H
    node_letter = lambda i: chr(64 + i)
    path_nodes = res["critical_nodes"]
    custom_crit = {(node_letter(a), node_letter(b)) for a, b in zip(path_nodes, path_nodes[1:])}

    G = nx.DiGraph()
    for u, v, w in CUSTOM_SCHEME_EDGES:
        G.add_edge(u, v, weight=w)

    pos = CUSTOM_POS
    edge_labels = {(u, v): str(w) for u, v, w in CUSTOM_SCHEME_EDGES}
    crit_edges = [(u, v) for u, v, w in CUSTOM_SCHEME_EDGES if (u, v) in custom_crit]
    other_edges = [(u, v) for u, v, w in CUSTOM_SCHEME_EDGES if (u, v) not in custom_crit]
    start_nodes_custom = ["A"]
    end_nodes_custom = ["H"]

    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    other_nodes_custom = [n for n in G.nodes() if n not in start_nodes_custom and n not in end_nodes_custom]
    nx.draw_networkx_nodes(
        G, pos, nodelist=other_nodes_custom, node_color=STYLE["node_color"], node_size=STYLE["node_size"],
        edgecolors=STYLE["edgecolors"], linewidths=STYLE["linewidths"], ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=start_nodes_custom, node_size=STYLE["node_size"],
        **START_NODE_STYLE, ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=end_nodes_custom, node_size=STYLE["node_size"],
        **END_NODE_STYLE, ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=STYLE["font_size"], font_color=STYLE["font_color"], ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color=STYLE["edge_color"], width=STYLE["edge_width"], ax=ax,
        arrows=True, arrowsize=STYLE["arrowsize"], arrowstyle=STYLE["arrowstyle"],
        min_source_margin=STYLE["arrow_margin"], min_target_margin=STYLE["arrow_margin"])
    nx.draw_networkx_edges(G, pos, edgelist=crit_edges, edge_color="crimson", width=STYLE["edge_width"], ax=ax,
        arrows=True, arrowsize=STYLE["arrowsize"], arrowstyle=STYLE["arrowstyle"],
        min_source_margin=STYLE["arrow_margin"], min_target_margin=STYLE["arrow_margin"])
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=STYLE["edge_label_font_size"], bbox=STYLE["edge_label_bbox"], ax=ax)

    path_str = " → ".join(node_letter(n) for n in res["critical_nodes"])
    ax.set_title(f"Собственный граф (один исток A). Начало (A) — зелёная рамка, окончание (H) — оранжевая. Критический путь: {path_str} — красный.", fontsize=STYLE["font_size"])
    ax.axis("off")
    plt.tight_layout()
    base = os.path.join(IMAGES_DIR, "graph_custom")
    plt.savefig(base + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Сохранено:", base + ".svg", "и", base + ".png")


def draw_graph(jobs, title, filename, critical_edges=None):
    """Сетевой граф в том же стиле, что и вариант 17; критический путь — красные рёбра."""
    res = run_cpm(jobs)
    G = res["G"]
    if critical_edges is None:
        critical_edges = res["critical_edges"]

    pos = nx.spring_layout(G, seed=42, k=1.8)
    edge_labels = {(u, v): f"{d['label']}({d['weight']})" for u, v, d in G.edges(data=True)}
    other_edges = [(u, v) for u, v in G.edges() if (u, v) not in critical_edges]

    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    nx.draw_networkx_nodes(
        G, pos, node_color=STYLE["node_color"], node_size=STYLE["node_size"],
        edgecolors=STYLE["edgecolors"], linewidths=STYLE["linewidths"], ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=STYLE["font_size"], font_color=STYLE["font_color"], ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color=STYLE["edge_color"], width=STYLE["edge_width"], ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=critical_edges, edge_color="crimson", width=STYLE["edge_width"], ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=STYLE["edge_label_font_size"], bbox=STYLE["edge_label_bbox"], ax=ax)

    ax.set_title(title, fontsize=STYLE["font_size"])
    ax.axis("off")
    plt.tight_layout()
    base = os.path.join(IMAGES_DIR, os.path.splitext(filename)[0])
    plt.savefig(base + ".svg", format="svg", bbox_inches="tight")
    plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Сохранено:", base + ".svg", "и", base + ".png")


def main():
    # Схема варианта 17: узлы A–F, цикл, веса на рёбрах
    draw_variant17_scheme()
    # Собственный граф: узлы A–H, веса на рёбрах, тот же стиль; критический путь красным
    draw_custom_scheme()


if __name__ == "__main__":
    main()
