import numpy as np
import matplotlib.pyplot as plt

k = -0.7608888888888889
N = 10000  

def generate_x():
    y = np.random.random()
    if y <= 0.144:
        return (12 * y)**(1/3) - 0.2
    else:
        return 2.5 - np.sqrt(2 * (y - 1) / k)

data = np.array([generate_x() for _ in range(N)])

x_theory = np.linspace(-0.2, 2.5, 1000)
f_theory = np.array([0.25*(x+0.2)**2 if x <= 1.0 else k*(x-2.5) for x in x_theory])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

y_points = np.random.uniform(0, 1, size=N)
# Оставляем только те точки, которые реально попадают под кривую f(x)
mask = []
for x in data:
    if x <= 1.0:
        mask.append(0.25*(x+0.2)**2)
    else:
        mask.append(k*(x-2.5))
y_points = y_points * np.array(mask)

ax1.plot(x_theory, f_theory, color='red', lw=2, label='Теория f(x)')
ax1.scatter(data, y_points, color='blue', s=0.1, alpha=0.3, label='Выборка (рассыпка)')
ax1.set_title('Область распределения точек (Monte-Carlo style)')
ax1.legend()

ax2.hist(data, bins=100, density=True, color='skyblue', alpha=0.7, label='Гистограмма выборки')
ax2.plot(x_theory, f_theory, 'r--', lw=2, label='Эталон')
ax2.set_title('Гистограмма (100 бинов для точности на x=1)')
ax2.legend()

plt.tight_layout()
plt.show()