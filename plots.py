import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ST.csv")
df_mean = df.groupby("fold", as_index=False).mean()

