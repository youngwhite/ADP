import copy
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ADP-esc50-ast.csv")
df.columns

# para, acc = df['para']/1e6, df['acc']*100
para = df.pivot(index='layers', columns='fold', values='para').mean(axis=1)/1e6
acc = df.pivot(index='layers', columns='fold', values='acc').mean(axis=1)*100
plt.plot(para, acc, marker='o')
plt.xlabel("Parameter Number (M)")
plt.ylabel("Average Accuracy (%)")
plt.title("Parameter Number vs. Average Accuracy")
plt.xticks(range(0, 100, 10))
plt.yticks(range(0, 101, 10))
plt.grid(True)

# 在坐标点旁边标记数值
for x, y in zip(para, acc):
    # plt.text(x, y, f'{y:.1f}', fontsize=12, ha='left', va='top')
    offset_x, offset_y = -3, 2
    plt.text(x + offset_x, y + offset_y, f'{y:.2f}', fontsize=9, ha='center', va='center')

plt.savefig("ADP-esc50-ast.png", dpi=300, bbox_inches='tight')  # PNG 格式，300 DPI
# plt.savefig("ADP-esc50-ast.pdf", bbox_inches='tight')  # PDF 格式，矢量图
plt.show()

# compare global optimization with self-distillation and without self-distillation
x = list(range(1, 13))
y = [6, 12.5, 36.25, 66, 78, 85.75, 90, 92.50, 94, 94.5, 94, 94.25]
y_sd = [4, 12.5, 39, 67.75, 81, 85, 90.75, 92.75, 93.75, 94.25, 94.5, 95.25]

# y = [4.5, 16.25, 35, 64.75, 74.75, 84.25, 88.75, 92.25, 93.5, 93.0, 94.5, 95.75]
# y_sd = [6.75, 13.25, 28.75, 57.25, 70, 82.75, 89.25, 92.25, 93.25, 93.75, 95.50, 95.75]
plt.plot(x, y, marker='o', label='Without SD')
plt.plot(x, y_sd, marker='o', label='With SD')
plt.legend()

