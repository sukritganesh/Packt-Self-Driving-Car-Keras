import numpy as np
import matplotlib.pyplot as plt

def draw(x1, x2):
    ln = plt.plot(x1, x2, '-')
    plt.pause(0.0001)
    ln[0].remove()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))

def calculate_error(line_parameters, points, y):
    p = sigmoid(points * line_parameters)
    m = points.shape[0]
    cross_entropy = -(1/m) * (np.log(p).T * y + np.log(1-p).T * (1 - y))
    return cross_entropy

def gradient_descent(line_parameters, points, y, alpha=0.001, epochs=500):
    m = points.shape[0]
    for i in range(epochs):
        p = sigmoid(points*line_parameters)
        gradient = (points.T * (p - y)) * (alpha/m)
        line_parameters = line_parameters - gradient
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        x1  = np.array(points[:, 0].min(), points[:, 0].max())
        x2 = -b / w2 + x1 * (-w1 / w2)
        draw(x1, x2)
        print('Error at Epoch', i, ':', calculate_error(line_parameters, points, y))

# MAIN CODE:

# blue points are positive, red points are negative
n_pts = 100
np.random.seed(0)
bias = np.ones(n_pts)
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T
all_points = np.vstack((top_region, bottom_region))

# Random initial weights
w1 = 0
w2 = 0
b = 0
line_parameters = np.matrix([w1, w2, b]).T

# Generate initial predictions
linear_combination = sigmoid(all_points*line_parameters)
probabilities = sigmoid(linear_combination)
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)

# Plot
_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
gradient_descent(line_parameters, all_points, y, alpha=0.01, epochs=500)
plt.show()
