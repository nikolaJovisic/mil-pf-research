import os
import time
import uuid
import logging
import torch
from datetime import timedelta

from extract_backbone import extract_backbone
from orchestrator import orchestrate_embeddings_and_training

import sys
sys.path.append('..')
from shim import *

CHECKPOINTS_DIR = f'{REPOS_DIR}/dinov2/outputs'
DESCRIPTION = 'Initial DINO orchestration.'

def setup_logger(log_path):
    logger = logging.getLogger("daemon_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        return logger

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def load_log(path):
    if not os.path.exists(path):
        return set()
    with open(path, 'r') as f:
        return set(line.strip() for line in f.readlines())

def mark_log(path, entry):
    with open(path, 'a') as f:
        f.write(entry + '\n')

def extract_duration_from_start(start_time):
    elapsed = timedelta(seconds=int(time.time() - start_time))
    hours, remainder = divmod(elapsed.seconds + elapsed.days * 86400, 3600)
    minutes = (remainder % 3600) // 60
    return f"{hours}h{minutes}m"

def extract_new_backbones(start_time, checkpoints_dir, models_dir, extracted_log, logger):
    extracted = load_log(extracted_log)
    new_paths = []

    folders = [
        f for f in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, f)) and f != 'archive'
    ]

    for folder in folders:
        folder_path = os.path.join(checkpoints_dir, folder)
        ckpt_file = os.path.join(folder_path, 'last_checkpoint.rank_0')

        if not os.path.exists(ckpt_file):
            continue

        with open(ckpt_file, 'r') as f:
            ckpt_name = f.read().strip()

        input_ckpt_path = os.path.join(folder_path, ckpt_name)
        if not os.path.exists(input_ckpt_path):
            continue

        duration_tag = extract_duration_from_start(start_time)
        output_name = f"{folder}_{duration_tag}.pt"
        output_path = os.path.join(models_dir, output_name)

        if output_path in extracted:
            continue

        logger.info(f"[{folder}] Extracting backbone → {output_name}")
        extract_backbone(input_ckpt_path, output_path)
        mark_log(extracted_log, output_path)
        logger.info(f"[{folder}] Extraction complete.")
        new_paths.append(output_path)

    return new_paths

def train_on_new_backbones(models_dir, trained_log, log_path, results_dir, logger):
    trained = load_log(trained_log)
    all_models = sorted([
        os.path.join(models_dir, f) for f in os.listdir(models_dir)
        if f.endswith('.pt') and os.path.isfile(os.path.join(models_dir, f))
    ])
    new_models = [m for m in all_models if m not in trained]

    if not new_models:
        logger.info("No new models to train on.")
        time.sleep(15 * 60)
        return

    num_gpus = torch.cuda.device_count()
    logger.info(f"Training on {len(new_models)} new extracted backbones using {num_gpus} GPUs.")

    orchestrate_embeddings_and_training(
        new_models,
        DESCRIPTION,
        log_path,
        results_dir
    )

    for model in new_models:
        mark_log(trained_log, model)
    logger.info("Training round completed.")

def loop_extract_and_train():
    daemon_id = str(uuid.uuid4())[:8]
    run_dir = f'{REPOS_DIR}/mammo_filter/orchestration/runs/{daemon_id}'
    os.makedirs(run_dir)

    extracted_log = os.path.join(run_dir, 'processed_backbones.log')
    trained_log = os.path.join(run_dir, 'trained_models.log')
    daemon_log = os.path.join(run_dir, 'daemon.log')
    models_dir = os.path.join(run_dir, 'models')

    logger = setup_logger(daemon_log)
    logger.info(f"=== Continuous extraction + training loop started === [daemon_id={daemon_id}]")

    start_time = time.time()

    while True:
        orchestrator_id = str(uuid.uuid4())[:8]
        base_dir = os.path.join(run_dir, orchestrator_id)

        log_path = os.path.join(base_dir, 'orchestrator.log')
        results_dir = os.path.join(base_dir, 'results')

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        new_backbones = extract_new_backbones(start_time, CHECKPOINTS_DIR, models_dir, extracted_log, logger)
        if new_backbones:
            logger.info(f"{len(new_backbones)} new backbones extracted.")

        train_on_new_backbones(models_dir, trained_log, log_path, results_dir, logger)

if __name__ == '__main__':
    loop_extract_and_train()
