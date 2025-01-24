import pandas as pd
import matplotlib.pyplot as plt

# ADP
df = pd.read_csv('ADP-esc50-ast.csv', index_col=None)

paras, accs = [], []
for i in range(1, 13):
    para, acc = df[df['layers'] == i]['para'].mean(), df[df['layers'] == i]['acc'].mean()
    paras.append(para/1e6)
    accs.append(acc)
plt.plot(paras, accs, marker='o')
plt.xlabel("Parameter Number (M)")
plt.ylabel("Average Accuracy (%)")
plt.title("Parameter Number vs. Average Accuracy")
plt.grid(True)

# short transformers 5-fold accs
result_short = {
    '79.14M': [0.9375, 0.9425, 0.945, 0.945, 0.925],
    '86.23M': [0.9475, 0.9475, 0.9375, 0.94, 0.93]
}

# LLM
