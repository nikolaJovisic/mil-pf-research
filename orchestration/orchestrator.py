import multiprocessing
import uuid
import logging
import os
import time
import torch
from embed_runner import run_embedding_pipeline
from grid_search import run_distributed_training


def setup_logger(log_path):
    logger = logging.getLogger()
    if logger.hasHandlers():
        return
    logger.setLevel(logging.INFO)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def orchestrate_embeddings_and_training(weights, description, logs_path):
    if log_path is None:
        log_path = f'{logs_path}/{uuid.uuid4()}.log'
    setup_logger(log_path)

    weights = [w.strip() for w in weights]
    
    num_gpus = torch.cuda.device_count()

    logging.info(f"=== ORCHESTRATION STARTED ===")
    logging.info(f"Log file: {log_path}")
    logging.info(f"Weights: {weights}")
    logging.info(f"Description: {description}")
    logging.info(f"Num GPUs: {num_gpus}")
    logging.info(f"=============================")

    if num_gpus < len(weights):
        raise ValueError("Number of GPUs must be at least equal to the number of weight files.")

    manager = multiprocessing.Manager()
    embedding_ids = manager.list()

    def run_embedding(weight_path, gpu_id, idx):
        logging.info(f"[EMBED {idx}] Starting embedding on GPU {gpu_id} with weights: {weight_path}")
        start_time = time.time()
        embedding_id = run_embedding_pipeline(
            weights=weight_path,
            description=description,
            gpu=gpu_id
        )
        duration = time.time() - start_time
        logging.info(f"[EMBED {idx}] Finished embedding (ID: {embedding_id}) on GPU {gpu_id} in {duration:.2f}s")
        embedding_ids.append(embedding_id)

    embedding_procs = []
    for i, weight in enumerate(weights):
        p = multiprocessing.Process(target=run_embedding, args=(weight, i, i))
        p.start()
        embedding_procs.append(p)

    for p in embedding_procs:
        p.join()

    for i, embedding_id in enumerate(embedding_ids):
        logging.info(f"[TRAIN {i}] Starting head training for embedding ID: {embedding_id} on {num_gpus} GPUs")
        start_time = time.time()
        run_distributed_training(embedding_id, num_gpus)
        duration = time.time() - start_time
        logging.info(f"[TRAIN {i}] Finished head training for embedding ID: {embedding_id} in {duration:.2f}s")

    logging.info(f"=== ORCHESTRATION COMPLETED ===")
