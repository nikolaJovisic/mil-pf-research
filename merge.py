import glob
import csv
import pandas as pd

output_file = 'results.csv'
input_files = sorted(glob.glob('results_gpu*.csv'))

with open(output_file, mode='w', newline='') as out_f:
    writer = None
    for i, file in enumerate(input_files):
        with open(file, mode='r', newline='') as in_f:
            reader = csv.reader(in_f)
            header = next(reader)
            if i == 0:
                writer = csv.writer(out_f)
                writer.writerow(header)
            for row in reader:
                writer.writerow(row)

def compute_cws(sensitivity, specificity):
    if sensitivity < 0.9 or specificity < 0.2:
        return 0
    return ((sensitivity - 0.9) / 0.1) + specificity

df = pd.read_csv("results.csv")
df["cws"] = df.apply(lambda row: compute_cws(row["sensitivity"], row["specificity"]), axis=1)
df["summed"] = df["sensitivity"] + df["specificity"]
df.to_csv("results.csv", index=False)

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

print(f"Merged {len(input_files)} files into {output_file} and saved non-dominated results to results_filtered.csv")
