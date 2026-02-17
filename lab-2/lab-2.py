import networkx as nx

def solve_cpm_full():
    G = nx.DiGraph()
    
    # Вариант 17
    jobs = [
        (1, 2, 0, 'A'),
        (2, 3, 5, 'B'),
        (2, 4, 3, 'C'),
        (3, 5, 6, 'D'),
        (4, 5, 4, 'E'),
        (5, 6, 5, 'F')
    ]
    
    for u, v, t, name in jobs:
        G.add_edge(u, v, weight=t, label=name)

    # 1. Прямой ход (Ранние сроки событий)
    es = {node: 0 for node in G.nodes()}
    for node in nx.topological_sort(G):
        for successor in G.successors(node):
            time = es[node] + G[node][successor]['weight']
            if time > es[successor]:
                es[successor] = time

    # 2. Обратный ход (Поздние сроки событий)
    total_time = max(es.values())
    lf = {node: total_time for node in G.nodes()}
    for node in reversed(list(nx.topological_sort(G))):
        for predecessor in G.predecessors(node):
            time = lf[node] - G[predecessor][node]['weight']
            if time < lf[predecessor]:
                lf[predecessor] = time

    # Вывод таблицы
    header = f"{'Работа':<7} | {'t':<2} | {'РН':<2} | {'РО':<2} | {'ПН':<2} | {'ПО':<2} | {'R (полн)':<7} | {'r (част)'}"
    print(header)
    print("-" * len(header))
    
    critical_nodes = [1]

    for u, v, d in G.edges(data=True):
        t = d['weight']
        name = d['label']
        
        rn = es[u]          # Раннее начало работы
        ro = rn + t         # Раннее окончание работы
        po = lf[v]          # Позднее окончание работы
        pn = po - t         # Позднее начало работы
        
        R_full = po - ro    # Полный резерв: ПО - РО
        r_priv = es[v] - ro # Частный резерв: Ранний срок след. узла - РО текущей работы
        
        # Фиксируем критический путь для вывода
        if R_full == 0 and u == critical_nodes[-1]:
            critical_nodes.append(v)

        print(f"{name+'('+str(u)+'-'+str(v)+')':<7} | {t:<2} | {rn:<2} | {ro:<2} | {pn:<2} | {po:<2} | {R_full:<7} | {r_priv}")

    print("-" * len(header))
    print(f"Критическое время проекта: {total_time}")
    print(f"Критический путь (узлы): {' -> '.join(map(str, critical_nodes))}")

if __name__ == "__main__":
    solve_cpm_full()