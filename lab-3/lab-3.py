"""
Лабораторная 3 (чистая версия):
- генерирует только 4 вариации параметров;
- по 5 графов на каждую вариацию (итого 20);
- пишет краткий отчет в images/interesting_4x5/README.md.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
OUT_DIR = os.path.join(IMAGES_DIR, "interesting_4x5")

N_POINTS = 100
SIDE = 100.0


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def generate_points(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, SIDE, size=(N_POINTS, 2))


def dist_matrix(pts):
    n = len(pts)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            D[i, j] = D[j, i] = d
    return D


def p_exp(d, a, b):
    if d <= 0:
        return 0.0
    return np.exp(-a * (d ** b))


def p_pow(d, b):
    d_safe = max(d, 1e-6)
    return 1.0 / (d_safe ** b)


def build_graph(D, n_edges, prob_fn, prob_arg, rng, max_degree_cap):
    n = D.shape[0]
    uf = UnionFind(n)
    degree = np.zeros(n, dtype=int)
    edges = []

    for _ in range(n_edges):
        candidates_i = []
        for i in range(n):
            if max_degree_cap is not None and degree[i] >= max_degree_cap:
                continue
            for j in range(n):
                if i != j and uf.find(i) != uf.find(j) and D[i, j] > 0:
                    candidates_i.append(i)
                    break
        if not candidates_i:
            break

        weights_i = np.array([degree[i] + 1 for i in candidates_i], dtype=float)
        weights_i /= weights_i.sum()
        i = rng.choice(candidates_i, p=weights_i)

        comp_i = uf.find(i)
        candidates_j = [j for j in range(n) if j != i and uf.find(j) != comp_i and D[i, j] > 0]
        if not candidates_j:
            continue

        probs = np.array([prob_fn(D[i, j], prob_arg) for j in candidates_j], dtype=float)
        s = probs.sum()
        if s <= 0:
            continue
        probs /= s
        j = candidates_j[int(rng.choice(len(candidates_j), p=probs))]

        uf.union(i, j)
        degree[i] += 1
        degree[j] += 1
        edges.append((i, j))

    return edges, degree


def mean_edge_length(edges, D):
    if not edges:
        return 0.0
    return sum(D[i, j] for i, j in edges) / len(edges)


def draw_graph(pts, edges, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pts[:, 0], pts[:, 1], s=15, c="steelblue", zorder=2)
    for i, j in edges:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], "k-", lw=0.4, alpha=0.7, zorder=1)
    ax.set_xlim(-5, SIDE + 5)
    ax.set_ylim(-5, SIDE + 5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def generate_interesting_4x5():
    presets = [
        {
            "id": "v1_exp_global",
            "title": "Вариация 1 (exp): экстремально глобальные связи и хабы",
            "formula": "exp",
            "a": 0.002,
            "b": 0.6,
            "n_edges": 99,
            "max_degree_cap": None,
            "looks_like": "Яркая сеть «центр–периферия»: много длинных рёбер и выраженные хабы.",
            "use_cases": "Транспортные сети с центральными узлами, иерархия филиалов, магистральные коммуникации.",
            "param_effect": "Очень малый a и b<1 дают слабое затухание по расстоянию.",
        },
        {
            "id": "v2_exp_local",
            "title": "Вариация 2 (exp): экстремально локальная сеть",
            "formula": "exp",
            "a": 1.2,
            "b": 3.8,
            "n_edges": 70,
            "max_degree_cap": 4,
            "looks_like": "Короткорёберная и разреженная сеть из соседей.",
            "use_cases": "Сенсорные сети, IoT на ограниченной территории, локальные mesh-сети.",
            "param_effect": "Большие a и b резко подавляют дальние рёбра.",
        },
        {
            "id": "v3_pow_hubs",
            "title": "Вариация 3 (1/d^b): дальнобойная разреженная сеть (почти цепочная)",
            "formula": "pow",
            "b": 0.12,
            "n_edges": 55,
            "max_degree_cap": 2,
            "looks_like": "Дальние связи есть, но структура вытянута в ветви и цепочки.",
            "use_cases": "Линейные/магистральные схемы, последовательные маршруты.",
            "param_effect": "Малый b сохраняет дальние рёбра, но cap=2 подавляет хабы.",
        },
        {
            "id": "v4_pow_local",
            "title": "Вариация 4 (1/d^b): сверхлокальная соседская сеть",
            "formula": "pow",
            "b": 7.0,
            "n_edges": 70,
            "max_degree_cap": 4,
            "looks_like": "Почти только связи с ближайшими соседями.",
            "use_cases": "Географически ограниченные сети доступа, локальные инженерные сети.",
            "param_effect": "Очень большой b делает дальние связи почти невозможными.",
        },
    ]

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    lines = [
        "# 4 вариации × 5 графов",
        "",
        "Сгенерировано 20 графов: по 5 на каждую вариацию.",
        "",
    ]

    for idx, cfg in enumerate(presets, 1):
        subdir = os.path.join(OUT_DIR, f"{idx}_{cfg['id']}")
        os.makedirs(subdir, exist_ok=True)

        avg_lens = []
        max_degs = []
        edge_counts = []

        formula_line = (
            f"- Формула: `exp(-a*d^b)`, `a={cfg['a']}`, `b={cfg['b']}`"
            if cfg["formula"] == "exp"
            else f"- Формула: `1/d^b`, `b={cfg['b']}`"
        )
        lines.extend([
            f"## {cfg['title']}",
            "",
            formula_line,
            f"- На что похоже: {cfg['looks_like']}",
            f"- Где применимо: {cfg['use_cases']}",
            f"- Связь с параметрами: {cfg['param_effect']}",
            "",
        ])

        for k in range(1, 6):
            seed_pts = 7000 + k * 13
            seed_rng = 9000 + idx * 100 + k * 29
            pts = generate_points(seed=seed_pts)
            D = dist_matrix(pts)
            rng = np.random.default_rng(seed_rng)

            if cfg["formula"] == "exp":
                a, b = cfg["a"], cfg["b"]
                prob_fn = lambda d, _: p_exp(d, a, b)
                prob_arg = None
                title = f"Вариация {idx}: exp(-a*d^b), a={a}, b={b}"
            else:
                b = cfg["b"]
                prob_fn = lambda d, x: p_pow(d, x)
                prob_arg = b
                title = f"Вариация {idx}: 1/d^b, b={b}"

            edges, degree = build_graph(D, cfg["n_edges"], prob_fn, prob_arg, rng, cfg["max_degree_cap"])
            avg_len = mean_edge_length(edges, D)

            filename = f"graph_{k}.png"
            save_path = os.path.join(subdir, filename)
            draw_graph(pts, edges, title, save_path)

            avg_lens.append(avg_len)
            max_degs.append(int(degree.max()) if len(degree) else 0)
            edge_counts.append(len(edges))

            lines.append(f"![{cfg['id']} #{k}](./{idx}_{cfg['id']}/{filename})")
            lines.append("")

        lines.extend([
            f"- По 5 графам: средняя длина ребра ≈ **{np.mean(avg_lens):.1f} ± {np.std(avg_lens):.1f}**;",
            f"  макс. степень ≈ **{np.mean(max_degs):.1f} ± {np.std(max_degs):.1f}**;",
            f"  число рёбер ≈ **{np.mean(edge_counts):.1f}**.",
            "",
            "---",
            "",
        ])

    out_md = os.path.join(OUT_DIR, "README.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Сгенерировано: {OUT_DIR}")
    print(f"Описание: {out_md}")


if __name__ == "__main__":
    generate_interesting_4x5()
