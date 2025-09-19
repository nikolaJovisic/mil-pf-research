import glob
import csv
import pandas as pd

def merge_csv_files(input_pattern, output_file):
    input_files = sorted(glob.glob(input_pattern))
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
    return input_files

def filter_non_dominated(input_file, output_file):
    metrics = ['auc', 'spec_w0.5', 'spec_w0.75', 'sens_w0.5', 'sens_w0.75', 'spec_90', 'spec_95', 'spec_100']
    df = pd.read_csv(input_file)
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
    filtered_df.to_csv(output_file, index=False)

def main():
    results_file = 'results.csv'
    filtered_results_file = 'results_filtered.csv'
    input_pattern = 'results/results_gpu*.csv'

    input_files = merge_csv_files(input_pattern, results_file)
    filter_non_dominated(results_file, filtered_results_file)

    print(f"Merged {len(input_files)} files into {results_file} and saved non-dominated results to {filtered_results_file}")

if __name__ == '__main__':
    main()
