import glob
import csv
import pandas as pd

results_file = 'results.csv'
input_files = sorted(glob.glob('results/imagenet/results_gpu*.csv'))

with open(results_file, mode='w', newline='') as out_f:
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

df = pd.read_csv(results_file)

metrics = ['auc', 'spec_w0.5', 'spec_w0.75', 'sens_w0.5', 'sens_w0.75', 'spec_90', 'spec_95', 'spec_100']

keep_indices = []
for i, row_i in df.iterrows():
    dominated = False
    for j, row_j in df.iterrows():
        if all(row_j[m] >= row_i[m] for m in metrics) and any(row_j[m] > row_i[m] for m in metrics):
            dominated = True
            break
    if not dominated:
        keep_indices.append(i)

filtered_df = df.loc[keep_indices].reset_index(drop=True)
filtered_df.to_csv("results_filtered.csv", index=False)

print(f"Merged {len(input_files)} files into {results_file} and saved non-dominated results to results_filtered.csv")
