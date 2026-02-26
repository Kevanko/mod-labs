"""
Лабораторная 3: генерация графов, похожих на реальные сети.
- 4 вариации параметров, по 5 графов на каждую (итого 20);
- графы сохраняются в images/interesting_4x5/.
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
            "param_effect": "Очень малый a и b<1 дают слабое затухание по расстоянию; искусственных ограничений степени нет (max_degree_cap=None).",
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
            "param_effect": "Большие a и b резко подавляют дальние рёбра; дополнительно введено искусственное ограничение степени вершины max_degree_cap=4.",
        },
        {
            "id": "v3_exp_medium",
            "title": "Вариация 3 (exp): средние по дальности связи",
            "formula": "exp",
            "a": 0.05,
            "b": 1.5,
            "n_edges": 80,
            "max_degree_cap": 6,
            "looks_like": "Умеренная смесь локальных и дальних рёбер, средние хабы.",
            "use_cases": "Региональные сети, смешанная топология.",
            "param_effect": "Средние a и b дают баланс между локальностью и дальними связями; искусственное ограничение степени max_degree_cap=6 не даёт узлам стать сверх-хабами.",
        },
        {
            "id": "v4_pow",
            "title": "Вариация 4 (1/d^b): степенная модель, варьируется b",
            "formula": "pow",
            "b": 1.2,
            "n_edges": 70,
            "max_degree_cap": 4,
            "looks_like": "Структура зависит от b: малый b — дальние связи, большой — локальная.",
            "use_cases": "Модели с степенным затуханием по расстоянию.",
            "param_effect": "В формуле 1/d^b варьируется только b; дополнительно введено искусственное ограничение степени max_degree_cap=4.",
        },
    ]

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    for idx, cfg in enumerate(presets, 1):
        subdir = os.path.join(OUT_DIR, f"{idx}_{cfg['id']}")
        os.makedirs(subdir, exist_ok=True)

        for k in range(1, 6):
            seed_pts = 7000 + k * 13
            seed_rng = 9000 + idx * 100 + k * 29
            pts = generate_points(seed=seed_pts)
            D = dist_matrix(pts)
            rng = np.random.default_rng(seed_rng)

            cap_str = (
                "без ограничения степени"
                if cfg["max_degree_cap"] is None
                else f"max_degree_cap={cfg['max_degree_cap']}"
            )

            if cfg["formula"] == "exp":
                a, b = cfg["a"], cfg["b"]
                prob_fn = lambda d, _: p_exp(d, a, b)
                prob_arg = None
                title = (
                    f"Вариация {idx}: P=exp(-a*d^b), a={a}, b={b}, "
                    f"{cap_str}, n_edges={cfg['n_edges']}"
                )
            else:
                b = cfg["b"]
                prob_fn = lambda d, x: p_pow(d, x)
                prob_arg = b
                title = (
                    f"Вариация {idx}: P=1/d^b, b={b}, "
                    f"{cap_str}, n_edges={cfg['n_edges']}"
                )

            edges, degree = build_graph(D, cfg["n_edges"], prob_fn, prob_arg, rng, cfg["max_degree_cap"])

            filename = f"graph_{k}.png"
            save_path = os.path.join(subdir, filename)
            draw_graph(pts, edges, title, save_path)

    print(f"Сгенерировано: {OUT_DIR}")


if __name__ == "__main__":
    generate_interesting_4x5()
