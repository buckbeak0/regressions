
# Linear Regression from Scratch 🧠📈

A simple implementation of **Linear Regression** using only Python — no fancy libraries, just pure logic. Inspired by [Neural Nine](https://www.youtube.com/@NeuralNine)'s educational content.

## 🔧 What You’ll Learn

- How Linear Regression works
- Implementing it from scratch
- Calculating error (MSE)
- Visualizing the regression line

---

## 📁 Project Structure

```
linear_regression_from_scratch/
├── main.py            # Main script containing the logic
├── data.csv           # Sample data file (X, Y)
└── README.md          # This file
```

---

## 📜 Requirements

Just Python, mate.

```bash
python3 main.py
```

---

## 🧮 How It Works

1. **Read the data** (CSV format: X and Y values)
2. **Calculate the slope (m) and intercept (b)** using the Least Squares method:
   ```
   m = ((mean(x) * mean(y)) - mean(x * y)) / ((mean(x))^2 - mean(x^2))
   b = mean(y) - m * mean(x)
   ```
3. **Predict values** with:  
   `y_pred = m * x + b`
4. **Calculate error** using **Mean Squared Error (MSE)**:
   ```
   MSE = sum((y - y_pred)^2) / n
   ```
5. **Visualize** the result using `matplotlib` (optional).

---

## 📊 Sample Output

```
Slope (m): 1.89
Intercept (b): 3.02
Mean Squared Error: 2.15
```

![plot](plot.png)

---

## 🚀 To Run

```bash
python3 main.py
```

Make sure `data.csv` is in the same directory and contains your X,Y values.

---

## 🎓 Inspired By

[Neural Nine](https://www.youtube.com/@NeuralNine) — go check him out if you want more brainy stuff with Python.

---

## 🧙‍♂️ License

Do what ye will, young wizard. This be free as a Hippogriff on a sunny day.

