import numpy as np
import matplotlib.pyplot as plt

# Константы (Вариант 7)
k = -0.7608888888888889
N = 20000

def generate_x():
    y = np.random.random()
    if y <= 0.144:
        return (12 * y)**(1/3) - 0.2
    else:
        return 2.5 - np.sqrt(2 * (y - 1) / k)

# 1. Генерация данных
data = np.array([generate_x() for _ in range(N)])

f_vals = np.where(data <= 1.0, 0.25*(data+0.2)**2, k*(data-2.5))
y_scatter = np.random.uniform(0, f_vals, size=N)

# 3. Теоретические кривые
x_theory = np.linspace(-0.2, 2.5, 1000)
f_theory = np.array([0.25*(x+0.2)**2 if x <= 1.0 else k*(x-2.5) for x in x_theory])
y_cdf_theory = [((0.25 * (xi + 0.2)**3) / 3 if xi <= 1.0 else 1 + (k * (xi - 2.5)**2) / 2) if xi > -0.2 else 0 for xi in x_theory]

# Отрисовка
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))
plt.subplots_adjust(hspace=0.4)

# График 1: Правильная рассыпка
ax1.plot(x_theory, f_theory, color='red', lw=3, label='Теория f(x)', zorder=3)
ax1.scatter(data, y_scatter, color='darkblue', s=0.05, alpha=0.1, label='Точки датчика (облако)')
ax1.set_title('1. Плотность распределения и визуализация плотности точек', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.2)

bins_edges = np.linspace(-0.2, 2.5, 81) # 80 интервалов, границы совпадут с 1.0

ax2.hist(data, bins=bins_edges, density=True, color='skyblue', 
         edgecolor='white', alpha=0.8, label='Результат генерации')
ax2.plot(x_theory, f_theory, 'r--', lw=2, label='Эталон f(x)')
ax2.set_title('2. Сравнение гистограммы с эталоном (проверка пика)', fontsize=12)
ax2.legend()

# График 3: CDF
ax3.plot(x_theory, y_cdf_theory, color='green', lw=3, label='Теория F(x)')
ax3.step(np.sort(data), np.linspace(0, 1, N), color='blue', alpha=0.4, label='Эмпирическая F(x)')
ax3.set_title('3. Функция распределения', fontsize=12)
ax3.legend()

plt.show()