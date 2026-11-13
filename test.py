import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

model = LogisticRegression().fit(X, y)
print("Accuracy:", model.score(X, y))

pd.DataFrame(X, columns=["a", "b"]).to_csv("data.csv", index=False)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig("plot.png")
print("All good")