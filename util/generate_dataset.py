import numpy as np
import matplotlib.pyplot as plt

def generate_quartic_data(n=80, seed=38):
    np.random.seed(seed)
    X = np.random.rand(n, 2)  # 각 샘플은 (x, y)

    def quartic_func(x):
        # 예시: y = -6x⁴ + 6x³ - x² + 0.2x + 0.4
        return -6 * x**4 + 6 * x**3 - 1 * x**2 + 0.2 * x + 0.4

    y = np.array([
        1 if point[1] > quartic_func(point[0]) else 0
        for point in X
    ])

    return X, y

# 시각화
if __name__ == "__main__":
    X, y = generate_quartic_data()

    plt.figure(figsize=(6, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')

    # 결정 경계 함수 시각화
    x_vals = np.linspace(0, 1, 300)
    y_vals = -6 * x_vals**4 + 6 * x_vals**3 - 1 * x_vals**2 + 0.2 * x_vals + 0.4
    plt.plot(x_vals, y_vals, color='black', linestyle='--', label='Decision Boundary')

    plt.legend()
    plt.title("Quartic (4차식) Decision Boundary")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()
