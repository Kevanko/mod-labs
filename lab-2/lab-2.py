"""
Лабораторная работа 2. Метод критического пути (CPM).
Два графа: по варианту 17 и свой (более сложный).
"""
import networkx as nx

# Граф по варианту 17 (одобренная таблица: A–B(5), A–C(3), B–D(6), C–E(4), D–F(5), E–F(5); Tкр=16, путь A→B→D→F)
GRAPH_VARIANT_17 = [
    (1, 2, 5, 'A'),   # A–B
    (1, 3, 3, 'B'),   # A–C
    (2, 4, 6, 'C'),   # B–D
    (3, 5, 4, 'D'),   # C–E
    (4, 6, 5, 'E'),   # D–F
    (5, 6, 5, 'F'),   # E–F
]

# Собственный граф по фото: A,B — старты; C(A,4), D(B,1), E(B,5), F(D,C,2), G(E,F,3), H(G,4)
# В нотации (i, j, t, name): A(1→3,4), B(2→4,1), C(2→5,5), D(3→6,2), E(4→6,2), F(5→7,3), G(6→7,3), H(7→8,4)
GRAPH_CUSTOM = [
    (1, 3, 4, 'A'),
    (2, 4, 1, 'B'),
    (2, 5, 5, 'C'),
    (3, 6, 2, 'D'),
    (4, 6, 2, 'E'),
    (5, 7, 3, 'F'),
    (6, 7, 3, 'G'),
    (7, 8, 4, 'H'),
]


def run_cpm(jobs):
    """
    Расчёт CPM по списку работ (u, v, t, name).
    Возвращает: es, lf, total_time, critical_nodes, critical_edges, rows.
    rows — список кортежей (name, u, v, t, rn, ro, pn, po, R_full, r_priv).
    """
    G = nx.DiGraph()
    for u, v, t, name in jobs:
        G.add_edge(u, v, weight=t, label=name)

    es = {node: 0 for node in G.nodes()}
    for node in nx.topological_sort(G):
        for successor in G.successors(node):
            time = es[node] + G[node][successor]['weight']
            if time > es[successor]:
                es[successor] = time

    total_time = max(es.values())
    lf = {node: total_time for node in G.nodes()}
    for node in reversed(list(nx.topological_sort(G))):
        for predecessor in G.predecessors(node):
            time = lf[node] - G[predecessor][node]['weight']
            if time < lf[predecessor]:
                lf[predecessor] = time

    rows = []
    critical_edges = set()
    for u, v, t, name in jobs:
        rn, ro = es[u], es[u] + t
        po, pn = lf[v], lf[v] - t
        R_full = po - ro
        r_priv = es[v] - ro
        rows.append((name, u, v, t, rn, ro, pn, po, R_full, r_priv))
        if R_full == 0:
            critical_edges.add((u, v))

    # Критический путь = самый длинный путь (макс. сумма длительностей), он задаёт минимальный срок проекта
    # Один критический путь (цепочка от 1 до последнего узла)
    order = list(nx.topological_sort(G))
    start, end = order[0], order[-1]
    critical_nodes = [start]
    cur = start
    while cur != end:
        for u, v in G.edges():
            if u == cur and (u, v) in critical_edges:
                critical_nodes.append(v)
                cur = v
                break
        else:
            break

    critical_path_names = [G[u][v]["label"] for u, v in zip(critical_nodes, critical_nodes[1:])]

    return {
        "es": es,
        "lf": lf,
        "total_time": total_time,
        "critical_nodes": critical_nodes,
        "critical_edges": critical_edges,
        "critical_path_names": critical_path_names,
        "rows": rows,
        "G": G,
    }

def solve_cpm_full(jobs=None):
    jobs = jobs or GRAPH_VARIANT_17
    res = run_cpm(jobs)
    node_letter = lambda i: chr(64 + i)
    header = f"{'Шифр':<7} | {'tij':<3} | {'РН':<2} | {'РО':<2} | {'ПН':<2} | {'ПО':<2} | {'Rij':<3} | {'rij'}"
    print(header)
    print("-" * len(header))
    for name, u, v, t, rn, ro, pn, po, R_full, r_priv in res['rows']:
        code = f"{node_letter(u)}-{node_letter(v)}"
        print(f"{code:<7} | {t:<3} | {rn:<2} | {ro:<2} | {pn:<2} | {po:<2} | {R_full:<3} | {r_priv}")
    print("-" * len(header))
    print(f"Критическое время проекта: {res['total_time']}")
    path_letters = " -> ".join(node_letter(n) for n in res["critical_nodes"])
    print(f"Критический путь: {path_letters}")



if __name__ == "__main__":
    print("========== Граф по варианту 17 ==========\n")
    solve_cpm_full(GRAPH_VARIANT_17)
    print("\n========== Свой граф (сложнее) ==========\n")
    solve_cpm_full(GRAPH_CUSTOM)