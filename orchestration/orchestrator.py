import subprocess
import uuid
import logging
import os
import time
import torch
import sys

sys.path.append('..')

def setup_logger(log_path):
    logger = logging.getLogger("orchestrator_logger")
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

def run_embedding_subprocess(weight_path, gpu_id, idx, description, logger):
    cmd = [
        sys.executable,
        "../embedding_inference/embed_runner.py",
        "--gpu", str(gpu_id),
        "--weights", weight_path,
        "--description", description
    ]

    logger.info(f"[EMBED {idx}] Launching subprocess with command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    return proc

def run_training_subprocess(embedding_id, model_id, results_dir, index, logger):
    cmd = [
        sys.executable,
        "../head_training/distributed_grid_search.py",
        "--embedding-id", embedding_id,
        "--model-id", model_id,
        "--results-dir", results_dir
    ]

    logger.info(f"[TRAIN {index}] Launching training subprocess with command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    for line in proc.stdout:
        logger.info(f"[TRAIN {index} STDOUT] {line.rstrip()}")

    proc.wait()

def orchestrate_embeddings_and_training(weights, description, log_path, results_dir):
    logger = setup_logger(log_path)

    weights = [w.strip() for w in weights]
    num_gpus = torch.cuda.device_count()

    logger.info(f"=== ORCHESTRATION STARTED ===")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Weights: {weights}")
    logger.info(f"Description: {description}")
    logger.info(f"Num GPUs: {num_gpus}")
    logger.info(f"=============================")

    if num_gpus < len(weights):
        raise ValueError("Number of GPUs must be at least equal to the number of weight files.")

    embedding_processes = []
    embedding_ids = [None] * len(weights)
    model_ids = [w.split('/')[-1].split('.')[0] for w in weights]

    for i, weight in enumerate(weights):
        proc = run_embedding_subprocess(weight, i, i, description, logger)
        embedding_processes.append((i, proc))

    for i, proc in embedding_processes:
        last_line = ""
        for line in proc.stdout:
            line = line.rstrip()
            logger.info(f"[EMBED {i} STDOUT] {line}")
            last_line = line
        proc.wait()
        embedding_ids[i] = last_line

    for i, (embedding_id, model_id) in enumerate(zip(embedding_ids, model_ids)):
        logger.info(f"[TRAIN {i}] Starting head training for embedding ID: {embedding_id} on {num_gpus} GPUs")
        start_time = time.time()
        run_training_subprocess(embedding_id, model_id, results_dir, i, logger)
        duration = time.time() - start_time
        logger.info(f"[TRAIN {i}] Finished head training for embedding ID: {embedding_id} in {duration:.2f}s")

    logger.info(f"=== ORCHESTRATION COMPLETED ===")
