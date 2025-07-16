import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import partial
from multiprocessing import Manager
from queue import Empty

import time

import pyrallis
import wandb
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record
from torch.cuda.comm import broadcast_coalesced
import torch.multiprocessing as mp

#from utils import evaluate_model_mm
#from dataset_utils import IndexBuffer

from grpo_mm import generate_rollout_data, grpo_loss, compute_log_probs
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


    sampler = DistributedSampler(train_data, num_replicas=config.num_reward_workers, rank=rank, shuffle=True)

    reward_fn = get_reward_function(config.failure_reward)
    step = 0

    dataloader = DataLoader(train_data, batch_size=config.batch_size // config.num_reward_workers, collate_fn=partial(collate_img_pc_v1, processor=processor, n_points=256), sampler=sampler,
                                num_workers=23, pin_memory=True)
    
    for epoch in range(config.train_epochs):

        print(f"Generator (Rank {rank}): Starting epoch {epoch + 1}/{config.train_epochs}.")
        sampler.set_epoch(epoch)

        for batch in dataloader:
            # synchronize the model parameters from GPU 1
            print(f"Generator (Rank {rank}): Synchronizing model parameters.")

            for param in model.parameters():
                dist.broadcast(param.data, src=config.num_reward_workers)

            print(f"Generating rollouts for batch {step + 1}/{len(dataloader)}")
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
            
        
            payload = {} 
            for key in [
                IPCKeys.INPUT_IDS,
                IPCKeys.ATT_MASK,
                IPCKeys.COMP_MASK,
                IPCKeys.ADV,
                IPCKeys.OLD_LOGP,
                IPCKeys.POINT_CLOUD,
                IPCKeys.IS_PC,
                IPCKeys.IS_IMG,
                IPCKeys.LOGITS_TO_KEEP,
                IPCKeys.AVG_REWARD

            ]:
                if key in rollout:
                    if isinstance(rollout[key], torch.Tensor):
                        payload[key] = rollout[key].detach().cpu().share_memory_()
                    else:
                        payload[key] = rollout[key]
            payload[IPCKeys.AVG_REWARD] = avg_reward
            queue.put(payload)
            step+=1

        # Signal to trainer that the epoch is finished
        queue.put(None)

    print(f"Generator (Rank {rank}): All epochs complete.")
    

def trainer_worker(queue, model, processor, config, rank):
    from torch.nn.utils.rnn import pad_sequence

    """GPU 1: compute loss, update model, evaluate."""
    print(f"Starting Trainer (Rank {rank}) ")
    torch.cuda.set_device(rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    step = 0

    reward_function = get_reward_function(config.failure_reward)

    loss_fn = partial(grpo_loss, processor=processor,epsilon_high=config.epsilon_high, epsilon_low=config.epsilon_low, reward_function=reward_function)

    
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
    )

    os.environ["WANDB_RUN_ID"] = run.id
    
    # Initial broadcast of parameters
    for param in model.parameters():
        dist.broadcast(param.data, src=rank)


    for epoch in range(config.train_epochs):
        end_signals = 0
        print(f"Trainer (Rank {rank}): Starting epoch {epoch + 1}/{config.train_epochs}.")
        # Keep stepping until all generators have signaled end‑of‑epoch

        while end_signals < config.num_reward_workers:
            payloads = []
            while len(payloads) <  config.num_reward_workers and end_signals < config.num_reward_workers:
                t0 = time.perf_counter()
                item = queue.get()
                wait = time.perf_counter() - t0 

                print(f"waiting time to get sample from queue {wait}")

                wandb.log({
                    "step": step,
                    "queue_wait":wait,

                })
                if item is None:
                    end_signals += 1
                    print(f" Received end signal ({end_signals}/{config.num_reward_workers})")
                else:
                    payloads.append(item)

            # if no payloads and everyone’s done, break out
            if not payloads:
                break

            for i, p in enumerate(payloads):
                print(f" after pad payload[{i}].input_ids.shape = {p['input_ids'].shape}")



            print(f"Trainer (Rank {rank}): received data from Generators")
            seq_keys = [
                IPCKeys.INPUT_IDS,
                IPCKeys.ATT_MASK,
                IPCKeys.COMP_MASK,
            ]

            # compute the max length along dim=1
            max_lens = {
                k: max(p[k].shape[1] for p in payloads)
                for k in seq_keys
            }

            for k in seq_keys:
                seqs = [p[k] for p in payloads]                          # list of [B_i, L_i]
                pad_val = processor.tokenizer.pad_token_id if k == IPCKeys.INPUT_IDS else 0
                padded = pad_sequence(seqs, batch_first=True, padding_value=pad_val)
                # padded shape: [num_payloads, max_lens[k], ...]
                for i, p in enumerate(payloads):
                    p[k] = padded[i]

            merged = {}

            for key in [
                IPCKeys.INPUT_IDS, IPCKeys.ATT_MASK, IPCKeys.COMP_MASK,
                IPCKeys.ADV, IPCKeys.POINT_CLOUD,
                IPCKeys.IS_PC, IPCKeys.IS_IMG,
            ]:
                merged[key] = torch.cat([p[key].to(rank, non_blocking=True)
                                         for p in payloads], dim=0)

            # optional video fields
            for opt_key in (IPCKeys.PIXEL_VALUES_VIDEOS, IPCKeys.VIDEO_GRID_THW):
                vals = [p.get(opt_key) for p in payloads if p.get(opt_key) is not None]
                merged[opt_key] = torch.cat(vals, dim=0) if vals else None


            merged[IPCKeys.LOGITS_TO_KEEP] = payloads[0][IPCKeys.LOGITS_TO_KEEP]

            batch_for_logprob = (
                merged[IPCKeys.INPUT_IDS],
                merged[IPCKeys.ATT_MASK],
                merged[IPCKeys.POINT_CLOUD],
                merged[IPCKeys.IS_PC],
                merged[IPCKeys.IS_IMG],
                merged.get(IPCKeys.PIXEL_VALUES_VIDEOS),
                merged.get(IPCKeys.VIDEO_GRID_THW),
            )
            # single unified compute
            old_lp = compute_log_probs(
                model,
                batch_for_logprob,
                merged[IPCKeys.LOGITS_TO_KEEP]
                ).detach()  # shape [big_batch, logits_to_keep]
            merged[IPCKeys.OLD_LOGP] = old_lp
            
            # keep logits_to_keep from the first payload

            # average reward
            avg_reward = sum(p[IPCKeys.AVG_REWARD] for p in payloads) / len(payloads)

            print(f"Trainer (Rank {rank}): received data from Generator")


            for grpo_iter in range(config.batch_updates):
                loss = loss_fn(model=model, rollout_data=merged)
                
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

                print(f"Epoch {epoch + 1}/{config.train_epochs}, Step {step + 1}, "
                        f"GRPO iter {grpo_iter + 1}/{config.batch_updates}, loss: {loss.item():.4f}", flush=True)
                
            del rollout_data
            torch.cuda.empty_cache()
           

            wandb.log({"average_reward": avg_reward, "step": step, "epoch": epoch + 1})
            print(f"Epoch {epoch + 1}, Step {step+1}, Avg Reward: {avg_reward:.4f}, Loss: {loss.item():.4f}", flush=True)
            step += 1  

            for param in model.parameters():
                dist.broadcast(param.data, src=rank)
                # Wait to receive from the generator
    if rank == config.num_reward_workers:
        wandb.finish()
    return




def main(
    rank: int, world_size: int, queue, config: TrainConfig):
    print(f"main invoked as rank={rank}, world_size={world_size}", flush=True)
    os.environ["RANK"]= str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"]    = str(world_size)

    setup(world_size)
    torch.cuda.set_device(rank)

    attn_implementation = 'flash_attention_2' if torch.cuda.is_available() else None
    print(f"Rank {rank}: Initializing model", flush=True)
    model = Cadrille.from_pretrained(
        config.sft_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map=rank).train().to(rank)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                              min_pixels=256 * 28 * 28,
                                              max_pixels=1280 * 28 * 28,
                                              padding_side="left",
                                              )

    eval_data_deepcad = RealDatasetMM(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_test', file_name='test.pkl', n_points=256, size=1000)
    eval_data_fusion = RealDatasetMM(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/fusion360_test', file_name='test.pkl', n_points=256, size=1000)
    train_data = RealDatasetMM(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_fusion_train', file_name=config.train_file, n_points=256, mode=config.train_mode, noise_scale_pc=0.01, size=config.train_size)
    print(f"Rank {rank}: Initializing datasets", flush=True)

    model = optimize_model_memory(model)

    #if rank == 1:
    #    model = DDP(model, device_ids=[rank], find_unused_parameters=True)


    print(f"\nRank {rank}: Starting RL fine-tuning using GRPO…")


    if rank < config.num_reward_workers:
        print(f"Rank {rank}: Starting reward inference worker", flush=True)
        reward_inference_worker(
            queue, model, processor, train_data, config, rank,
        )
    else:
        print(f"Rank {rank}: Starting trainer worker", flush=True)
        trainer_worker(
            queue, model, processor, config, rank,
        )
    cleanup()
    print("Training completed.")

@pyrallis.wrap()
def spawn_main(config: TrainConfig):

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1245"
    
    # 1) parse config exactly once
    world_size = config.num_reward_workers + 1
    queue = mp.Queue(maxsize=2*config.num_reward_workers )
    mp.spawn(
        fn=main,
        nprocs=world_size,
        args=(world_size, queue, config,),
        join=True,
    )

if __name__ == "__main__":

    mp.set_start_method("spawn")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,7"
    spawn_main()
