import os
import time
import uuid
import logging
import torch
from datetime import timedelta

from extract_backbone import extract_backbone
from orchestrator import orchestrate_embeddings_and_training

daemon_id = uuid.uuid4()

CHECKPOINTS_DIR = '/home/nikola.jovisic.ivi/nj/dinov2/outputs'
MODELS_DIR = f'/home/nikola.jovisic.ivi/nj/mammo_filter/orchestration/runs/{daemon_id}/models'
LOGS_DIR = f'/home/nikola.jovisic.ivi/nj/mammo_filter/orchestration/runs/{daemon_id}/logs'
EXTRACTED_LOG = f'{LOGS_DIR}/processed_backbones.log'
TRAINED_LOG = f'{LOGS_DIR}/trained_models.log'

DESCRIPTION = 'Initial DINO orchestration.'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{LOGS_DIR}/daemon_{daemon_id}.log"),
            logging.StreamHandler()
        ]
    )

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

def extract_new_backbones(start_time):
    extracted = load_log(EXTRACTED_LOG)
    new_paths = []

    folders = [
        f for f in os.listdir(CHECKPOINTS_DIR)
        if os.path.isdir(os.path.join(CHECKPOINTS_DIR, f)) and f != 'archive'
    ]

    for folder in folders:
        folder_path = os.path.join(CHECKPOINTS_DIR, folder)
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
        output_path = os.path.join(MODELS_DIR, output_name)

        if output_path in extracted:
            continue

        logging.info(f"[{folder}] Extracting backbone → {output_name}")
        extract_backbone(input_ckpt_path, output_path)
        mark_log(EXTRACTED_LOG, output_path)
        logging.info(f"[{folder}] Extraction complete.")
        new_paths.append(output_path)

    return new_paths

def train_on_new_backbones():
    trained = load_log(TRAINED_LOG)
    all_models = sorted([
        os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR)
        if f.endswith('.pt') and os.path.isfile(os.path.join(MODELS_DIR, f))
    ])
    new_models = [m for m in all_models if m not in trained]

    if not new_models:
        logging.info("No new models to train on.")
        return

    num_gpus = torch.cuda.device_count()
    logging.info(f"Training on {len(new_models)} new extracted backbones using {num_gpus} GPUs.")
    orchestrate_embeddings_and_training(new_models, DESCRIPTION, num_gpus, log_path=f"{LOGS_DIR}/orchestrator_{uuid.uuid4()}.log")
    for model in new_models:
        mark_log(TRAINED_LOG, model)
    logging.info("Training round completed.")

def loop_extract_and_train():
    setup_logger()
    logging.info(f"=== Continuous extraction + training loop started === [daemon_id={daemon_id}]")
    start_time = time.time()

    while True:
        new_backbones = extract_new_backbones(start_time)
        if new_backbones:
            logging.info(f"{len(new_backbones)} new backbones extracted.")
        train_on_new_backbones()

if __name__ == '__main__':
    loop_extract_and_train()
