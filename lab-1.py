import numpy as np
import matplotlib.pyplot as plt

k = -0.7608888888888889
N = 1000  # Количество точек

def generate_x():
    y = np.random.random()
    if y <= 0.144:
        return (12 * y)**(1/3) - 0.2
    else:
        return 2.5 - np.sqrt(2 * (y - 1) / k)

# Генерация данных
data = [generate_x() for _ in range(N)]
x_theory = np.linspace(-0.2, 2.5, 500)
f_theory = [0.25*(x+0.2)**2 if x <= 1 else k*(x-2.5) for x in x_theory]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Плотность и точки (Scatter)
ax1.plot(x_theory, f_theory, 'r', lw=2, label='Плотность f(x)')
ax1.scatter(data, np.zeros_like(data), color='blue', alpha=0.1, label='Точки на оси X')
ax1.set_title('1. Плотность и нагенерированные точки')
ax1.legend()
ax1.grid()

# График 2: Плотность и Гистограмма (Bounds check)
ax2.hist(data, bins=30, density=True, alpha=0.4, color='skyblue', label='Гистограмма')
ax2.plot(x_theory, f_theory, 'r', lw=2, label='f(x)')
ax2.set_title('2. Проверка границ и формы распределения')
ax2.set_xlim([-0.5, 3.0]) # Показываем, что за пределы [-0.2, 2.5] не выходит
ax2.legend()
ax2.grid()

plt.show()