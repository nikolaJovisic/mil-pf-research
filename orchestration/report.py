import os
import sys
import glob
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from merge import merge_csv_files, filter_non_dominated


def get_latest_run_id(runs_dir='runs'):
    subdirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    if not subdirs:
        raise RuntimeError("No subfolders found in 'runs/'")
    subdirs.sort(key=lambda d: os.path.getctime(os.path.join(runs_dir, d)), reverse=True)
    return subdirs[0]


def extract_model_and_time(name):
    time_match = re.search(r'(\d+)h(\d+)m', name)
    base_model = re.sub(r'_\d+h\d+m', '', name)
    hours = int(time_match.group(1)) if time_match else 0
    minutes = int(time_match.group(2)) if time_match else 0
    return base_model, hours * 60 + minutes


def analyze_model_performance_to_pdf(csv_path, pdf_path):
    df = pd.read_csv(csv_path)
    df[['base_model', 'train_minutes']] = df['model_name'].apply(
        lambda x: pd.Series(extract_model_and_time(x))
    )

    # If there is a baseline row, duplicate it for each base_model as 0 min
    baseline_df = df[df['model_name'] == 'baseline'].copy()
    base_models = df['base_model'].unique()
    baseline_augmented = []

    for model in base_models:
        if model != 'baseline' and not df[(df['base_model'] == model) & (df['train_minutes'] == 0)].empty:
            continue
        temp = baseline_df.copy()
        temp['base_model'] = model
        temp['train_minutes'] = 0
        baseline_augmented.append(temp)

    if baseline_augmented:
        df = pd.concat([df] + baseline_augmented, ignore_index=True)

    with PdfPages(pdf_path) as pdf:
        for model in df['base_model'].unique():
            if model == 'baseline':
                continue

            model_df = df[df['base_model'] == model].copy()

            max_auc_df = model_df.groupby('train_minutes')['auc'].max().reset_index()
            max_auc_df = max_auc_df.sort_values('train_minutes')

            mean_top10_auc_df = (
                model_df.sort_values(['train_minutes', 'auc'], ascending=[True, False])
                .groupby('train_minutes')
                .head(10)
                .groupby('train_minutes')['auc']
                .mean()
                .reset_index()
                .sort_values('train_minutes')
            )

            plt.figure(figsize=(10, 6))
            sns.lineplot(data=max_auc_df, x='train_minutes', y='auc', label='Max AUC')
            sns.lineplot(data=mean_top10_auc_df, x='train_minutes', y='auc', label='Mean Top-10 AUC')
            plt.title(f'Performance of Model: {model}')
            plt.xlabel('Training Time (minutes)')
            plt.ylabel('AUC')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()



def collect_and_merge_csvs(run_id):
    base_path = os.path.join('runs', run_id)
    output_dir = '.'
    merged_csv_path = os.path.join(output_dir, f'results_{run_id}.csv')
    filtered_csv_path = os.path.join(output_dir, f'results_filtered_{run_id}.csv')

    all_rows = []

    # Step 1: Traverse grid search results
    for eval_iter in os.listdir(base_path):
        eval_iter_path = os.path.join(base_path, eval_iter)
        if not os.path.isdir(eval_iter_path) or eval_iter == 'models':
            continue
        results_path = os.path.join(eval_iter_path, 'results')
        if not os.path.isdir(results_path):
            continue
        for model_name in os.listdir(results_path):
            model_path = os.path.join(results_path, model_name)
            if not os.path.isdir(model_path):
                continue
            csv_pattern = os.path.join(model_path, 'results_gpu*.csv')
            temp_output = os.path.join(output_dir, f'_temp_{run_id}_{eval_iter}_{model_name}.csv')
            input_files = merge_csv_files(csv_pattern, temp_output)
            if not input_files:
                continue
            df = pd.read_csv(temp_output)
            df.insert(0, 'model_name', model_name)
            all_rows.append(df)
            os.remove(temp_output)

    # Step 2: Add baseline if exists
    baseline_path = os.path.join(output_dir, 'baseline_results.csv')
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        baseline_df.insert(0, 'model_name', 'baseline')
        all_rows.append(baseline_df)
        print("Included baseline_results.csv")

    # Step 3: Save merged and filtered files
    if not all_rows:
        print("No CSV files found.")
        return

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df.to_csv(merged_csv_path, index=False)
    filter_non_dominated(merged_csv_path, filtered_csv_path)

    print(f"Saved merged results to: {merged_csv_path}")
    print(f"Saved filtered non-dominated results to: {filtered_csv_path}")

    # Step 4: Generate PDF with AUC performance plots
    pdf_path = os.path.join(output_dir, f'report_{run_id}.pdf')
    analyze_model_performance_to_pdf(merged_csv_path, pdf_path)
    print(f"Saved AUC performance plots to: {pdf_path}")


if __name__ == '__main__':
    if len(sys.argv) == 2:
        run_id = sys.argv[1]
    else:
        run_id = get_latest_run_id()
        print(f"No run_id provided. Using latest run folder: {run_id}")
    collect_and_merge_csvs(run_id)
