# test_time.py
import time
import torch
from functools import partial
from torch.utils.data import DataLoader

import pyrallis
from transformers import AutoProcessor

from grpo_mm import generate_rollout_data
from train_cadrille_grpo import (
    TrainConfig,
    collate_img_pc_v1,
    get_reward_function,
    optimize_model_memory,
)
from cad_recode_model_mm import Cadrille
from dataset_utils import RealDatasetMM
from utils import init_pool


@pyrallis.wrap()
def main(config: TrainConfig):
    # force use of GPU 0
    torch.cuda.set_device(7)
    device = torch.device("cuda:7")

    # load & optimize model
    model = Cadrille.from_pretrained(
        config.sft_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        device_map=7,
    )
    model = optimize_model_memory(model).to(device).eval()

    # load processor
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side="left",
    )

    train_data = RealDatasetMM(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_fusion_train', file_name=config.train_file, n_points=256, mode=config.train_mode, noise_scale_pc=0.01, size=config.train_size)


    reward_fn = get_reward_function(config.failure_reward)

    # try a few batch sizes (you can adjust this list)
    batch_sizes = [config.batch_size //  config.num_reward_workers,
                   config.batch_size,
                   config.batch_size * 2]
    
    for bs in batch_sizes:

        print(f"Starting timing tests for batch_size {bs}")

        loader = DataLoader(
            train_data,
            batch_size=bs,
            collate_fn=partial(collate_img_pc_v1, processor=processor, n_points=256),
            num_workers=5,
            pin_memory=True,
        )
        batch = next(iter(loader))
        init_pool(18)
        # warm‑up
        print(f"warming up")
        torch.cuda.synchronize(device)
        _ = generate_rollout_data(
            model, reward_fn, processor, batch,
            config.num_generations,
            config.max_completion_length,
            top_samples=config.top_samples,
            gpg=config.use_gpg,
            buffer=None,
        )

        # timing
        start = time.perf_counter()
        rollout, avg_reward = generate_rollout_data(
            model, reward_fn, processor, batch,
            config.num_generations,
            config.max_completion_length,
            top_samples=config.top_samples,
            gpg=config.use_gpg,
            buffer=None,
        )

        batch = next(iter(loader))

        rollout, avg_reward = generate_rollout_data(
            model, reward_fn, processor, batch,
            config.num_generations,
            config.max_completion_length,
            top_samples=config.top_samples,
            gpg=config.use_gpg,
            buffer=None,
        )
        elapsed = time.perf_counter() - start

        print(f"[Batch size {bs:4d}]  Time: {elapsed:.3f}s    Avg reward: {avg_reward:.4f}")


if __name__ == "__main__":
    main()
