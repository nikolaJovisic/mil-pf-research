import pandas as pd

df = pd.read_csv("results.csv")

keep_indices = []
for i, row_i in df.iterrows():
    dominated = False
    for j, row_j in df.iterrows():
        if (
            row_j['specificity'] > row_i['specificity']
            and row_j['sensitivity'] > row_i['sensitivity']
        ):
            dominated = True
            break
    if not dominated:
        keep_indices.append(i)

filtered_df = df.loc[keep_indices].reset_index(drop=True)
filtered_df.to_csv("results_filtered.csv", index=False)

