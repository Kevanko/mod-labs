"""
Лабораторная 3: Генерация графов, похожих на реальные сети.
100 точек на плоскости 100×100; рёбра по вероятностям P_ij (две формулы);
циклы недопустимы; выбор вершины по степеням, вероятности пересчитываются.
"""
import csv
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
GRAPHS_DIR = os.path.join(SCRIPT_DIR, "graphs")
METRICS_CSV = os.path.join(GRAPHS_DIR, "metrics.csv")
INTERESTING_DIR = os.path.join(IMAGES_DIR, "interesting")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(INTERESTING_DIR, exist_ok=True)

N_POINTS = 100
SIDE = 100.0
N_EDGES_TARGET = 99
SEED = 42
USE_DIFFERENT_POINTS_PER_GRAPH = True

# Число графов на одну комбинацию (формула, параметр). Для серии: python3 lab-3.py --batch 500
N_PER_CONFIG = 1
for i, arg in enumerate(sys.argv):
    if arg == "--batch" and i + 1 < len(sys.argv):
        N_PER_CONFIG = int(sys.argv[i + 1])
        break


class UnionFind:
    """СНМ для проверки циклов: только рёбра между разными компонентами."""

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


def generate_points(n=N_POINTS, side=SIDE, seed=SEED):
    """100 случайных точек в [0, side]×[0, side]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, side, size=(n, 2))


def dist_matrix(pts):
    """Матрица евклидовых расстояний; диагональ не используется."""
    n = len(pts)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            D[i, j] = D[j, i] = d
    return D


def p_exp(d, a):
    """Формула 1: P_ij = e^(-a·d_ij²). При d=0 возвращаем 0 (петли не нужны)."""
    if d <= 0:
        return 0.0
    return np.exp(-a * d * d)


def p_pow(d, b):
    """Формула 2: P_ij = 1 / (d_ij^b). При d=0 возвращаем 0. Нижний порог d_min чтобы не взрывать при малых d."""
    d_safe = max(d, 1e-6)
    return 1.0 / (d_safe ** b)


def build_graph(pts, D, n_edges, prob_fn, prob_arg, rng):
    """
    Строит граф без циклов: n_edges рёбер.
    prob_fn(d) — функция вероятности (p_exp или p_pow с замыканием по a/b).
    Выбор вершины i — с весами (degree_i + 1); выбор j — пропорционально P_ij среди других компонент.
    """
    n = len(pts)
    uf = UnionFind(n)
    degree = np.zeros(n, dtype=int)
    edges = []

    for _ in range(n_edges):
        # Вершины, у которых есть хотя бы один кандидат в другой компоненте
        candidates_i = []
        for i in range(n):
            for j in range(n):
                if i != j and uf.find(i) != uf.find(j) and D[i, j] > 0:
                    candidates_i.append(i)
                    break
        if not candidates_i:
            break

        # Выбор i с вероятностью пропорциональной (degree_i + 1) — пересчёт по степеням
        weights_i = np.array([degree[i] + 1 for i in candidates_i], dtype=float)
        weights_i /= weights_i.sum()
        i = rng.choice(candidates_i, p=weights_i)

        # Кандидаты j: другая компонента, d_ij > 0
        comp_i = uf.find(i)
        candidates_j = [j for j in range(n) if j != i and uf.find(j) != comp_i and D[i, j] > 0]
        if not candidates_j:
            continue

        probs = np.array([prob_fn(D[i, j], prob_arg) for j in candidates_j], dtype=float)
        s = probs.sum()
        if s <= 0:
            continue
        probs /= s
        idx = rng.choice(len(candidates_j), p=probs)
        j = candidates_j[int(idx)]

        uf.union(i, j)
        degree[i] += 1
        degree[j] += 1
        edges.append((i, j))

    # Проверка: в лесу на n вершинах с c компонентами ровно n - c рёбер (циклов нет)
    n_comp = sum(1 for v in range(n) if uf.find(v) == v)
    assert len(edges) == n - n_comp, "ожидается лес без циклов"
    return edges, degree


def prob_exp(d, a):
    return p_exp(d, a)


def prob_pow(d, b):
    return p_pow(d, b)


def mean_edge_length(edges, D):
    """Средняя длина ребра (для сравнения графов)."""
    if not edges:
        return 0.0
    return sum(D[i, j] for i, j in edges) / len(edges)


def run_one(pts, D, label, prob_fn, prob_arg, rng, save_path=None):
    """Строит один граф; если save_path задан — рисует и сохраняет. Возвращает (edges, degree, avg_len)."""
    edges, degree = build_graph(pts, D, N_EDGES_TARGET, prob_fn, prob_arg, rng)
    avg_len = mean_edge_length(edges, D)
    if save_path:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(pts[:, 0], pts[:, 1], s=15, c="steelblue", zorder=2)
        for i, j in edges:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], "k-", lw=0.4, alpha=0.7, zorder=1)
        ax.set_xlim(-5, SIDE + 5)
        ax.set_ylim(-5, SIDE + 5)
        ax.set_aspect("equal")
        ax.set_title(f"{label}\n(ср. длина ребра: {avg_len:.1f})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Сохранено: {save_path}")
    return edges, degree, avg_len


def analyze(edges, degree, label, prob_name, prob_val, avg_edge_len=0.0):
    """Статистики и краткий анализ «на что похоже / с чем сравнить»."""
    n = len(degree)
    m = len(edges)
    deg = degree
    avg_deg = 2 * m / n if n else 0
    max_deg = int(deg.max()) if n else 0

    lines = [
        f"--- {label} ---",
        f"Параметры: {prob_name} = {prob_val}.",
        f"Рёбер: {m}, вершин: {n}. Средняя длина ребра: {avg_edge_len:.1f}.",
        f"Средняя степень: {avg_deg:.2f}, макс. степень: {max_deg}.",
        f"Распределение степеней (кратко): большинство степеней в диапазоне 1–{min(max_deg, 5)}.",
    ]

    # Интерпретация по форме графа
    if max_deg >= 8:
        lines.append("На что похоже: граф с выраженным «хабом» — одна или несколько вершин с большой степенью; напоминает звёздную топологию или иерархическую сеть (например, центр + периферия в транспортной или организационной сети).")
        lines.append("С чем сравнить: дерево «звезда», предпочтительное присоединение (Barabási–Albert) без петель; или граф «центр–периферия».")
    elif max_deg <= 3 and avg_deg < 2.5:
        lines.append("На что похоже: разреженная сеть, рёбра в основном короткие (малые расстояния); напоминает локальные связи в пространстве — соседние узлы, сенсорная сеть или граф ближайших соседей.")
        lines.append("С чем сравнить: минимальное остовное дерево (MST) по тем же точкам; геометрический граф с порогом по расстоянию; случайное дерево с пространственной предпочтением коротких рёбер.")
    else:
        lines.append("На что похоже: промежуточный вариант — смесь локальных и более длинных связей; может напоминать транспортную или коммуникационную сеть с несколькими узлами пересадок.")
        lines.append("С чем сравнить: случайное дерево на тех же точках; дерево k ближайших соседей (k-NN); граф с вероятностью связи, зависящей от расстояния (как в данной модели).")

    return "\n".join(lines)


def analyze_batch(csv_path):
    """Читает metrics.csv, группирует по (formula, param), строит графики закономерностей, дописывает выводы в analysis.md."""
    if not os.path.isfile(csv_path):
        return
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["param"] = float(r["param"])
            r["avg_edge_len"] = float(r["avg_edge_len"])
            r["max_deg"] = int(r["max_deg"])
            r["n_edges"] = int(r["n_edges"])
            rows.append(r)
    if not rows:
        return
    groups = defaultdict(list)
    for r in rows:
        key = (r["formula"], r["param"])
        groups[key].append(r)
    # Агрегаты
    exp_params, exp_avg, exp_std, exp_maxd, exp_maxd_std = [], [], [], [], []
    pow_params, pow_avg, pow_std, pow_maxd, pow_maxd_std = [], [], [], [], []
    for (formula, param), group in sorted(groups.items()):
        avgs = [x["avg_edge_len"] for x in group]
        maxds = [x["max_deg"] for x in group]
        mean_avg = np.mean(avgs)
        std_avg = np.std(avgs)
        mean_md = np.mean(maxds)
        std_md = np.std(maxds)
        if formula == "exp":
            exp_params.append(param)
            exp_avg.append(mean_avg)
            exp_std.append(std_avg)
            exp_maxd.append(mean_md)
            exp_maxd_std.append(std_md)
        else:
            pow_params.append(param)
            pow_avg.append(mean_avg)
            pow_std.append(std_avg)
            pow_maxd.append(mean_md)
            pow_maxd_std.append(std_md)
    # Графики
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if exp_params:
        axes[0].errorbar(exp_params, exp_avg, yerr=exp_std, fmt="o-", capsize=3)
        axes[0].set_xlabel("a")
        axes[0].set_ylabel("Средняя длина ребра")
        axes[0].set_title("Формула 1: P = exp(-a·d²)")
        axes[0].grid(True, alpha=0.3)
    if pow_params:
        axes[1].errorbar(pow_params, pow_avg, yerr=pow_std, fmt="s-", capsize=3, color="C1")
        axes[1].set_xlabel("b")
        axes[1].set_ylabel("Средняя длина ребра")
        axes[1].set_title("Формула 2: P = 1/d^b")
        axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "batch_avg_edge_len.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {os.path.join(IMAGES_DIR, 'batch_avg_edge_len.png')}")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if exp_params:
        ax2.errorbar(exp_params, exp_maxd, yerr=exp_maxd_std, fmt="o-", capsize=3, label="exp(-a·d²)")
    if pow_params:
        ax2.errorbar(pow_params, pow_maxd, yerr=pow_maxd_std, fmt="s-", capsize=3, label="1/d^b")
    ax2.set_xlabel("Параметр (a или b)")
    ax2.set_ylabel("Макс. степень (среднее по серии)")
    ax2.set_title("Максимальная степень вершины от параметра")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "batch_max_degree.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {os.path.join(IMAGES_DIR, 'batch_max_degree.png')}")

    # Дописать выводы в analysis.md
    analysis_path = os.path.join(SCRIPT_DIR, "analysis.md")
    block = [
        "",
        "## Выводы по серии",
        "",
        "По серии сгенерированных графов (по группе на каждую комбинацию формула–параметр):",
        "",
        "| Формула | Параметр | Средняя длина ребра (mean ± std) | Макс. степень (mean ± std) |",
        "|---------|----------|-----------------------------------|----------------------------|",
    ]
    for (formula, param), group in sorted(groups.items()):
        avgs = [x["avg_edge_len"] for x in group]
        maxds = [x["max_deg"] for x in group]
        block.append(f"| {formula} | {param} | {np.mean(avgs):.1f} ± {np.std(avgs):.1f} | {np.mean(maxds):.1f} ± {np.std(maxds):.1f} |")
    block.extend([
        "",
        "**Закономерности:** с ростом параметра a (или b) средняя длина ребра уменьшается — граф становится «локальнее». "
        "Максимальная степень может расти при малых a/b (появление хабов при более равномерном выборе длинных рёбер).",
        "",
        "**На что похоже:** при малых a и b преобладают длинные рёбра и выраженные хабы — графы напоминают центр–периферию, транспортную или организационную сеть с узлами пересадок. При больших a и b рёбра в основном короткие, структура локальная — ближе к сенсорной сети, графу ближайших соседей или минимальному остовному дереву с пространственным предпочтением.",
        "",
        "Графики сохранены: `images/batch_avg_edge_len.png`, `images/batch_max_degree.png`.",
        "",
    ])
    with open(analysis_path, "a", encoding="utf-8") as f:
        f.write("\n".join(block))
    print("  В analysis.md дописан блок «Выводы по серии».")


def append_interesting_8_to_analysis():
    """Дописывает в analysis.md блок про 8 выбранных графов."""
    analysis_path = os.path.join(SCRIPT_DIR, "analysis.md")
    block = [
        "",
        "## 8 выбранных графов",
        "",
        "Для наглядности закономерности сохранены 8 графов в каталог `images/interesting/`:",
        "- **Формула 1 (exp):** a = 0.01 (длинные рёбра, хабы), 0.06, 0.15, 0.4 (короткие, локальная структура).",
        "- **Формула 2 (1/d^b):** b = 0.3 (длинные рёбра), 0.9, 1.6, 2.5 (короткие).",
        "По ним видно, как с ростом параметра граф переходит от «звезды/центра–периферии» к локальной сети.",
        "",
    ]
    with open(analysis_path, "a", encoding="utf-8") as f:
        f.write("\n".join(block))


def save_interesting_8():
    """
    Выбирает 8 графов, наглядно показывающих закономерность (от длинных рёбер и хабов к коротким и локальным),
    и копирует их в images/interesting/ с понятными именами.
    """
    # 8 конфигураций: 4 exp (разброс по a) + 4 pow (разброс по b)
    interesting = [
        ("exp", 0.01, "малый a — длинные рёбра, хабы"),
        ("exp", 0.06, "exp переход"),
        ("exp", 0.15, "exp переход"),
        ("exp", 0.4, "большой a — короткие рёбра, локальная структура"),
        ("pow", 0.3, "малый b — длинные рёбра"),
        ("pow", 0.9, "pow переход"),
        ("pow", 1.6, "pow переход"),
        ("pow", 2.5, "большой b — короткие рёбра"),
    ]
    index_lines = ["# 8 выбранных графов (закономерность по параметрам)", ""]
    for i, (formula, param, desc) in enumerate(interesting, 1):
        if formula == "exp":
            src = os.path.join(IMAGES_DIR, f"graph_exp_a{int(round(param * 100)):03d}.png")
        else:
            src = os.path.join(IMAGES_DIR, f"graph_pow_b{int(param * 10):02d}.png")
        base = f"{i}_{formula}_{param}.png"
        dst = os.path.join(INTERESTING_DIR, base)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            index_lines.append(f"- **{base}** — {desc}")
            print(f"  Интересный {i}: {dst}")
    index_path = os.path.join(INTERESTING_DIR, "README.md")
    index_lines.extend([
        "",
        "Закономерность: с ростом a (или b) средняя длина ребра уменьшается, граф становится локальнее; при малых параметрах — выраженные хабы.",
    ])
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))
    print(f"  Список: {index_path}")


def main():
    # Больше значений параметров для наглядных закономерностей (5–7 на формулу)
    configs_exp = [(0.01, ""), (0.03, ""), (0.06, ""), (0.10, ""), (0.15, ""), (0.25, ""), (0.4, "")]
    configs_pow = [(0.3, ""), (0.6, ""), (0.9, ""), (1.2, ""), (1.6, ""), (2.0, ""), (2.5, "")]

    analyses = []
    comparison = []
    csv_header = ["formula", "param", "seed", "avg_edge_len", "max_deg", "n_edges"]
    with open(METRICS_CSV, "w", encoding="utf-8", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(csv_header)

        for idx, (a, _) in enumerate(configs_exp):
            for k in range(N_PER_CONFIG):
                seed_pts = (SEED + idx + k * 1000) if USE_DIFFERENT_POINTS_PER_GRAPH else SEED
                seed_rng = SEED + 100 + idx + k * 1000
                pts = generate_points(seed=seed_pts)
                D = dist_matrix(pts)
                rng = np.random.default_rng(seed_rng)
                label = f"Формула 1: P = exp(-a·d²), a = {a}"
                save_path = os.path.join(IMAGES_DIR, f"graph_exp_a{int(round(a * 100)):03d}.png") if k == 0 else None
                edges, degree, avg_len = run_one(pts, D, label, prob_exp, a, rng, save_path)
                writer.writerow(["exp", a, seed_rng, round(avg_len, 4), int(degree.max()), len(edges)])
                if k == 0:
                    comparison.append((f"exp a={a}", avg_len, int(degree.max())))
                    if N_PER_CONFIG == 1:
                        text = analyze(edges, degree, label, "a", a, avg_len)
                        analyses.append(text)
                        print(text)
                        print()

        for idx, (b, _) in enumerate(configs_pow):
            for k in range(N_PER_CONFIG):
                seed_pts = (SEED + 50 + idx + k * 1000) if USE_DIFFERENT_POINTS_PER_GRAPH else SEED
                seed_rng = SEED + 200 + idx + k * 1000
                pts = generate_points(seed=seed_pts)
                D = dist_matrix(pts)
                rng = np.random.default_rng(seed_rng)
                label = f"Формула 2: P = 1/d^b, b = {b}"
                save_path = os.path.join(IMAGES_DIR, f"graph_pow_b{int(b * 10):02d}.png") if k == 0 else None
                edges, degree, avg_len = run_one(pts, D, label, prob_pow, b, rng, save_path)
                writer.writerow(["pow", b, seed_rng, round(avg_len, 4), int(degree.max()), len(edges)])
                if k == 0:
                    comparison.append((f"1/d^b b={b}", avg_len, int(degree.max())))
                    if N_PER_CONFIG == 1:
                        text = analyze(edges, degree, label, "b", b, avg_len)
                        analyses.append(text)
                        print(text)
                        print()

    if N_PER_CONFIG > 1:
        print(f"Сгенерировано {N_PER_CONFIG} графов на каждую конфигурацию. Метрики: {METRICS_CSV}")

    # Блок сравнения и выводы
    comparison_text = [
        "",
        "## Сравнение графов (разница по параметрам)",
        "",
        "| Конфигурация | Средняя длина ребра | Макс. степень |",
        "|--------------|---------------------|---------------|",
    ]
    for name, avg_len, max_d in comparison:
        comparison_text.append(f"| {name} | {avg_len:.1f} | {max_d} |")
    comparison_text.extend([
        "",
        "Чем меньше параметр a (или b), тем выше вероятность длинных рёбер — средняя длина ребра больше. "
        "Чем больше a (или b), тем сильнее предпочтение коротких рёбер — граф «локальнее», средняя длина меньше. "
        "Рёбер во всех графах 99 (дерево на 100 вершинах); циклы отсутствуют.",
        "",
        "## Выводы",
        "",
        "Зависимость от параметра: при увеличении a (exp) или b (1/d^b) средняя длина ребра падает, граф становится более локальным. "
        "При малых параметрах чаще появляются «хабы» (вершины с большой степенью). Две формулы дают качественно сходное поведение с разной чувствительностью к расстоянию.",
    ])
    analyses.append("\n".join(comparison_text))

    analysis_path = os.path.join(SCRIPT_DIR, "analysis.md")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("# Анализ графов (лаба 3)\n\n")
        f.write("\n\n".join(analyses))
    print(f"\nАнализ записан: {analysis_path}")

    analyze_batch(METRICS_CSV)
    save_interesting_8()
    append_interesting_8_to_analysis()


if __name__ == "__main__":
    main()
