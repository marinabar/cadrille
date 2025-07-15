import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import partial
from multiprocessing import Manager
from queue import Empty

import pyrallis
import wandb
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record

from utils import evaluate_model_mm
from dataset_utils import IndexBuffer

from grpo_mm import generate_rollout_data, grpo_loss
from train_cadrille_grpo import TrainConfig, collate_img_pc_v1, get_reward_function, optimize_model_memory, setup, cleanup

from cad_recode_model_mm import Cadrille

from transformers import AutoProcessor
from dataset_utils import RealDatasetMM

@dataclass
# class to hold IPC keys, that will be transferred between processes
class IPCKeys:
    INPUT_IDS: str = "input_ids"
    ATT_MASK: str = "attention_mask"
    COMP_MASK: str = "completion_mask"
    ADV: str = "advantages"
    OLD_LOGP: str  = "old_log_probs"    
    POINT_CLOUD: str = "point_cloud"
    IS_PC: str = "is_pc"
    IS_IMG: str = "is_img"
    PIXEL_VALUES_VIDEOS: str = "pixel_values_videos"
    VIDEO_GRID_THW: str = "video_grid_thw"
    AVG_REWARD: str = "avg_reward"
    LOGITS_TO_KEEP: str = "logits_to_keep"


def reward_inference_worker(queue, model, processor, train_data, config, rank):
    """GPU 0: sample rollouts, compute old log‑probs & advantages, enqueue minimal tensors."""
    torch.cuda.set_device(rank)
    sampler = DistributedSampler(train_data, num_replicas=2, rank=rank, shuffle=True)
    loader  = DataLoader(
        train_data,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=partial(collate_img_pc_v1, processor=processor, n_points=256),
        num_workers=24,
        pin_memory=True,
        drop_last=True,
    )
    reward_fn = get_reward_function(config.failure_reward)

    for epoch in range(config.train_epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            # synchronize the model parameters from GPU 1
            for param in model.parameters():
                dist.broadcast(param.data, src=1)

            rollout, avg_reward = generate_rollout_data(
                model,
                reward_fn,
                processor,
                batch,
                config.num_generations,
                config.max_completion_length,
                top_samples=config.top_samples,
                gpg=config.use_gpg,
                buffer = None)
            
            payload = {
                IPCKeys.INPUT_IDS: rollout["input_ids"].cpu(),
                IPCKeys.ATT_MASK: rollout["attention_mask"].cpu(),
                IPCKeys.COMP_MASK: rollout["completion_mask"].cpu(),
                IPCKeys.ADV: rollout["advantages"].cpu(),
                IPCKeys.OLD_LOGP: rollout["old_log_probs"].cpu(),
                IPCKeys.POINT_CLOUD: rollout["point_cloud"].cpu(),
                IPCKeys.IS_PC: rollout["is_pc"].cpu(),
                IPCKeys.IS_IMG: rollout["is_img"].cpu(),
                IPCKeys.LOGITS_TO_KEEP: rollout["logits_to_keep"],
                IPCKeys.AVG_REWARD: avg_reward,
            }
            # Handle optional video tensors
            if rollout.get("pixel_values_videos") is not None:
                payload[IPCKeys.PIXEL_VALUES_VIDEOS] = rollout["pixel_values_videos"].cpu()
                payload[IPCKeys.VIDEO_GRID_THW] = rollout["video_grid_thw"].cpu()

            queue.put(payload)

        # Signal to trainer that the epoch is finished
        queue.put(None)
    print(f"Generator (Rank {rank}): All epochs complete.")

def trainer_worker(queue, model, processor, config, rank, run_id):

    """GPU 1: compute loss, update model, evaluate."""
    torch.cuda.set_device(rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    wandb.init(id=run_id, resume="allow")

    step = 0
    loss_fn = partial(grpo_loss, processor=processor,epsilon_high=config.epsilon_high, epsilon_low=config.epsilon_low)

    if rank == 1:
        wandb.init(id=run_id, resume="allow", project=config.project, group=config.group, name=config.name)

    for epoch in range(config.train_epochs):
        print(f"Trainer (Rank {rank}): Starting epoch {epoch + 1}/{config.train_epochs}.")
        while True:
            for param in model.module.parameters():
                dist.broadcast(param.data, src=rank)
                # Wait to receive from the generator

            payload = queue.get()
            if payload is None:
                print(f"Trainer (Rank {rank}): Received end-of-epoch signal.")
                break  # End of epoch

            rollout_data = {
                key: (val.to(rank, non_blocking=True) if isinstance(val, torch.Tensor) else val)
                for key, val in payload.items()
            }
            avg_reward = rollout_data.pop(IPCKeys.AVG_REWARD, None)

            for grpo_iter in range(config.batch_updates):
                loss = loss_fn(model=model, rollout_data=rollout_data)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

                wandb.log({
                    "loss": loss.item(),
                    "step": step,
                    "grpo_iter": grpo_iter + 1,
                    "epoch": epoch + 1,
                })            

            wandb.log({"average_reward": avg_reward, "step": step, "epoch": epoch + 1})
            print(f"Epoch {epoch + 1}, Step {step+1}, Avg Reward: {avg_reward:.4f}, Loss: {loss.item():.4f}")
            step += 1  

    if rank == 1:
        wandb.finish()
    return




@record
@pyrallis.wrap()
def main(config: TrainConfig):

    rank = int(os.environ.get("LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))

    assert world_size == 2, "Use --nproc_per_node=2"

    setup(world_size)
    torch.cuda.set_device(rank)

    mgr= Manager()
    queue = mgr.Queue(maxsize=2)

    attn_implementation = 'flash_attention_2' if torch.cuda.is_available() else None

    model = Cadrille.from_pretrained(
        config.sft_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map=rank).train().to(rank)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                              min_pixels=256 * 28 * 28,
                                              max_pixels=1280 * 28 * 28,
                                              padding_side="left")

    eval_data_deepcad = RealDatasetMM(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_test', file_name='test.pkl', n_points=256, size=1000)
    eval_data_fusion = RealDatasetMM(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/fusion360_test', file_name='test.pkl', n_points=256, size=1000)
    train_data = RealDatasetMM(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_fusion_train', file_name=config.train_file, n_points=256, mode=config.train_mode, noise_scale_pc=0.01, size=config.train_size)


    model = optimize_model_memory(model)


    print("\nStarting RL fine-tuning using GRPO...")
    training_config = {
        'train_epochs': config.train_epochs,
        'batch_size': config.batch_size,  # reduce if you have fewer GPUs
        'num_generations': config.num_generations,  # reduce if you have GPUs with less VRAM
        'top_samples': config.top_samples,  # reduce if you have GPUs with less VRAM
        'max_completion_length': config.max_completion_length,  # reduce if you have GPUs with less VRAM
        'learning_rate': config.learning_rate,
        'batch_updates': config.batch_updates,
        'epsilon_high': config.epsilon_high,
        'epsilon_low': config.epsilon_low,
    }
    

    if rank == 0:
        reward_inference_worker(
            queue, model, processor, train_data, config, rank
        )
    else:
        run = wandb.init(
            project=config.project,
            group=config.group,
            name=config.name,
            config=asdict(config),
            reinit=True,
        )
        run_id = run.id
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        trainer_worker(
            queue, model, processor, config, rank, run_id
        )
    cleanup()
    print("Training completed.")

if __name__ == "__main__":
    main()
        

