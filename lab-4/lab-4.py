"""
Лабораторная 4: Выборки с возвращением и без, размещение m=n.
Матрица m×n: строка i — i-й выбор, столбец j — элемент j; в ячейке 1, если выбран.
"""
import numpy as np

SEED = 42


def with_replacement(n, m, rng):
    """Выборка с возвращением: m раз выбираем один из n (индексы 0..n-1)."""
    mat = np.zeros((m, n), dtype=int)
    for i in range(m):
        j = rng.integers(0, n)
        mat[i, j] = 1
    return mat


def without_replacement(n, m, rng):
    """Выборка без возвращения: m различных из n."""
    perm = rng.permutation(n)[:m]
    mat = np.zeros((m, n), dtype=int)
    for i, j in enumerate(perm):
        mat[i, j] = 1
    return mat


def placement_m_n(n, rng):
    """Размещение m=n: перестановка всех n элементов."""
    perm = rng.permutation(n)
    mat = np.zeros((n, n), dtype=int)
    for i, j in enumerate(perm):
        mat[i, j] = 1
    return mat


def print_table(title, n, m, mat):
    """Печать таблицы с подписями строк 1..m и столбцов 1..n."""
    print(title)
    print(f"n = {n}, m = {m}")
    rows, cols = mat.shape
    head = "   " + " ".join(f"{j+1:>3}" for j in range(cols))
    print(head)
    print("   " + "-" * (4 * cols))
    for i in range(rows):
        print(f"{i+1:>2}|" + " ".join(f"{mat[i,j]:>3}" for j in range(cols)))
    print()


def main():
    rng = np.random.default_rng(SEED)

    # n, m в диапазоне 3..10

    # С возвращением: 4 таблицы
    for n, m in [(5, 4), (7, 5), (8, 6), (6, 4)]:
        mat = with_replacement(n, m, rng)
        print_table("--- Выборка С ВОЗВРАЩЕНИЕМ ---", n, m, mat)

    # Без возвращения: 4 таблицы
    for n, m in [(6, 4), (8, 5), (7, 4), (5, 3)]:
        mat = without_replacement(n, m, rng)
        print_table("--- Выборка БЕЗ ВОЗВРАЩЕНИЯ ---", n, m, mat)

    # Размещение m=n: 4 таблицы
    for n in [5, 7, 8, 4]:
        mat = placement_m_n(n, rng)
        print_table("--- РАЗМЕЩЕНИЕ m = n ---", n, n, mat)


if __name__ == "__main__":
    main()
