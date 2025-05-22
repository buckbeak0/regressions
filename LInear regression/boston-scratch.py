import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

boston = fetch_openml(name='boston', version=1, as_frame=True)
df = pd.DataFrame(boston.frame)

def gradient_descent(m_now, b_now, points, x_col, y_col, L):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i][x_col]
        y = points.iloc[i][y_col]
        
        m_gradient += (-2/n) * x * (y - (m_now * x + b_now))
        b_gradient += (-2/n) * (y - (m_now * x + b_now))
        
    m = m_now - m_gradient * L 
    b = b_now - b_gradient * L
    
    return m, b

def compute_mse(m, b, points, x_col, y_col):
    total_error = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i][x_col]
        y = points.iloc[i][y_col]
        y_pred = m * x + b
        total_error += (y - y_pred) ** 2
    return total_error / n

data = df[['RM', 'MEDV']].copy()

m = 0
b = 0
L = 0.01
epochs = 0 
mse_history = []
mse = compute_mse(m, b, data, 'RM', 'MEDV')

while mse > 45:
    m, b = gradient_descent(m, b, data, x_col='RM', y_col='MEDV', L=L)
    mse = compute_mse(m, b, data, 'RM', 'MEDV')
    mse_history.append(mse)
    epochs += 1
    if epochs % 100 == 0:
        print(f"Epoch = {epochs}: m = {m:.4f}, b = {b:.4f}, MSE = {mse:.4f}")


plt.scatter(data.RM, data.MEDV, color = 'blue', alpha=0.5, label = 'Data')

x_vals = data.RM
y_vals = m * x_vals + b
plt.plot(x_vals, y_vals, color = 'darkred', label = 'Gradient Descent Line')

plt.xlabel('Average number of roooms (RM)')
plt.ylabel('Median value of rooms (MEDV in 1000$s)')
plt.legend()
plt.title('Linear Regression using gradient Descent')
plt.grid(True)

plt.figure(figsize=(8,5))
plt.plot(range(epochs), mse_history, color='purple')
plt.title("MSE vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.show()