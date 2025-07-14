import pandas as pd

def compute_cws(sensitivity, specificity):
    if sensitivity < 0.9 or specificity < 0.2:
        return 0
    return ((sensitivity - 0.9) / 0.1) + specificity

df = pd.read_csv("results.csv")
df["cws"] = df.apply(lambda row: compute_cws(row["sensitivity"], row["specificity"]), axis=1)
df["summed"] = df.apply(lambda row: row["sensitivity"] + row["specificity"], axis=1)
df.to_csv("results.csv", index=False)
