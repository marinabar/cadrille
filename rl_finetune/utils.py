import concurrent
import multiprocessing
import os
import random


os.environ["PYGLET_HEADLESS"] = "True"

from collections import defaultdict
from argparse import ArgumentParser
from multiprocessing import Process, Queue
from multiprocessing.pool import Pool
from functools import partial

class NonDaemonProcess(Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NonDaemonPool(Pool):
    def Process(self, *args, **kwargs):
        proc = super(NonDaemonPool, self).Process(*args, **kwargs)
        proc.__class__ = NonDaemonProcess
        return proc

# CAD recode imports
import numpy as np
import torch

import cadquery as cq

import trimesh
from scipy.spatial import cKDTree

from torch.utils.data import DataLoader
from tqdm import tqdm

from normal_consistency import compute_normals_metrics

def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None

    Explanation:
        1. Sets seed for Python's built-in random module for basic random operations.
        2. Sets seed for NumPy, ensuring consistent random number generation in array operations.
        3. Sets seed for PyTorch CPU operations.
        4. If CUDA is available, sets seed for all GPU devices.
        5. Configures cuDNN to ensure deterministic behavior:
           - Sets deterministic flag to True, ensuring reproducible results.
           - Disables benchmarking to prevent algorithm selection based on hardware.

    Note:
        Setting deterministic behavior may impact performance but ensures consistent results
        across multiple runs, which is crucial for debugging and research.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_metrics_from_single_text(text, gt_file, pred_mesh_path, pred_brep_path, n_points):

    gt_file = os.path.abspath(gt_file)
    pred_mesh_dir = os.path.abspath(pred_mesh_path)
    pred_brep_dir = os.path.abspath(pred_brep_path)

    base_file = os.path.basename(gt_file).rsplit('.stl', 1)[0]

    print(f"computing metrics for file: {gt_file}", flush=True)
    #print(f"saving temp mesh to: {pred_mesh_dir}", flush=True)

    mesh_path = os.path.abspath(os.path.join(pred_mesh_dir, base_file + '.stl'))
    brep_path = os.path.abspath(os.path.join(pred_brep_dir, base_file + '.step'))

    # runs the python code and saves the mesh and brep files
    pred_mesh = code_to_mesh_and_brep_less_safe(text, mesh_path, brep_path)

    if pred_mesh is None:
        print("Skipping metrics: invalid prediction mesh", flush=True)
        return dict(file_name=base_file, cd=None, iou=None, auc=None, mean_cos=None)

    cd, iou, auc, mean_cos = None, None, None, None
    try:  # apply_transform fails for some reason; or mesh path can not exist
        gt_mesh = trimesh.load_mesh(gt_file)

        gt_mesh = transform_gt_mesh(gt_mesh)
        
        #print("Loaded and normalized ground truth", flush=True)
        
        pred_mesh = transform_pred_mesh(pred_mesh)
        #print("Normalizing prediction", flush=True)

        

        cd = compute_cd(gt_mesh, pred_mesh, n_points)
        iou = compute_iou(gt_mesh, pred_mesh)
        auc, mean_cos, _ = compute_normals_metrics(
                pred_mesh, gt_mesh, tol=2
            )
        print(f"CD {cd} IoU {iou} AUC {auc} Mean Cos {mean_cos}", flush=True)


    except Exception as e:
        pass
    
    return dict(file_name=base_file, cd=cd, iou=iou, auc=auc, mean_cos=mean_cos)


from multiprocessing import get_context
def get_metrics_from_texts(texts, meshes, max_workers=None):

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    """
    Processes a list of texts in parallel, assigning a unique id to each to avoid temp file conflicts.

    Args:
        texts (list of str): The generated texts to process.
        meshes (list of str): The paths to the meshes corresponding to each text.
        max_workers (int, optional): Number of parallel workers.

    Returns:
        list: A list of meshes (or None for failed processes).
    """

    temp_path = "./tmp_data"
    pred_mesh_path = os.path.join(temp_path, 'tmp_mesh')
    pred_brep_path = os.path.join(temp_path, 'tmp_brep')

    os.makedirs(pred_mesh_path, exist_ok=True)
    os.makedirs(pred_brep_path, exist_ok=True)

    n_points = 8192
    results = []
    if max_workers is None:
        max_workers = os.cpu_count()

    ctx = get_context("spawn")
    
    with NonDaemonPool(processes=max_workers, context=ctx) as pool:
        results = list(tqdm(pool.starmap(
            partial(
                get_metrics_from_single_text,
                pred_mesh_path=pred_mesh_path,
                pred_brep_path=pred_brep_path,
                n_points=n_points),
                zip(texts, meshes)), total=len(meshes)))

    return results



def compute_iou(gt_mesh, pred_mesh):
    try:
        intersection_volume = 0
        for gt_mesh_i in gt_mesh.split():
            for pred_mesh_i in pred_mesh.split():
                intersection = gt_mesh_i.intersection(pred_mesh_i)
                volume = intersection.volume if intersection is not None else 0
                intersection_volume += volume
        
        gt_volume = sum(m.volume for m in gt_mesh.split())
        pred_volume = sum(m.volume for m in pred_mesh.split())
        union_volume = gt_volume + pred_volume - intersection_volume
        assert union_volume > 0
        return intersection_volume / union_volume
    except:
        pass


def compute_cd(pred_mesh, gt_mesh, n_points=8192):
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    cd = np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))
    return cd


def compute_metrics(pred_mesh, gt_mesh):
    return compute_cd(pred_mesh, gt_mesh), compute_iou(pred_mesh, gt_mesh)


def evaluate_model_mm(model, processor, eval_examples, device, collate_fn, batch_size=8):
    model.eval()
    print("\n" + "=" * 50)
    print("EVALUATION ON", len(eval_examples), "EXAMPLES")
    print("=" * 50)

    dataloader = DataLoader(eval_examples, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    ious, cds = [], []
    n_incorrect, n_failed_intersect = 0, 0

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            generated_ids = model.generate(input_ids=batch['input_ids'].to(model.device),
                                           attention_mask=batch['attention_mask'].to(model.device),
                                           point_clouds=batch['point_clouds'].to(model.device),
                                           is_pc=batch['is_pc'].to(model.device),
                                           is_img=batch['is_img'].to(model.device),
                                           pixel_values_videos=batch['pixel_values_videos'].to(
                                               model.device) if batch.get('pixel_values_videos',
                                                                          None) is not None else None,
                                           video_grid_thw=batch['video_grid_thw'].to(model.device) if batch.get(
                                               'video_grid_thw', None) is not None else None,
                                           max_new_tokens=768,
                                           bad_words_ids=[[model.config.video_token_id]],
                                           )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
            ]
            py_strings = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            decoded_texts = py_strings
            pred_metrics = get_metrics_from_texts(decoded_texts, batch["mesh_path"], max_workers=24)
            for metrics in pred_metrics:
                if metrics is None or metrics["iou"] is None:
                    n_incorrect += 1
                    continue
                if metrics["iou"] < 0:
                    n_failed_intersect += 1
                else:
                    ious.append(metrics["iou"])
                    cds.append(metrics["cd"])

    print(f"IoU mean {np.mean(ious)}, median {np.median(ious)}")
    print(f"CD mean {np.mean(cds)}, median {np.median(cds)}")
    print(f"Invalid generations fraction: {n_incorrect / len(eval_examples)}")
    print(f"Intersect failure fraction: {n_failed_intersect / len(eval_examples)}")
    print("=" * 50)

    model.train()
    return ious, cds, n_incorrect / len(eval_examples), n_failed_intersect / len(eval_examples)

def transform_real_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
    mesh.apply_scale(2.0 / max(mesh.extents))  # Normalize to [-1, 1]
    return mesh

def transform_gt_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
    extent = np.max(mesh.extents)
    if extent > 1e-7:
            mesh.apply_scale(1.0 / extent)
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh



def transform_pred_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_scale(1.0 / 200)  # Normalize to [0, 1]
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh


########## evaluate functions ##########
def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def code_to_mesh_and_brep(code_str, mesh_path, brep_path):


    #print(f"executing code {code_str}")
    # saves mesh and brep files from code string 
    try:
        ns = safe_ns.copy()
        exec(code_str, ns)
        compound = ns['r'].val()
        mesh = compound_to_mesh(compound)
        assert len(mesh.faces) > 2
        mesh.export(mesh_path)
        print("mesh exported successfully")
        # cq.exporters.export(compound, brep_path)
    except Exception as e:
        print("error executing the python code and exporting the mesh:", e)
        return


safe_ns = {"cq": cq}

def code_to_mesh_and_brep_less_safe(code_str, mesh_path, brep_path):
    ns=safe_ns.copy()
    #print(f"Executing code {code_str}")
    try:
        exec(code_str, ns)
        mesh = compound_to_mesh(ns["r"].val())
        # export files if needed
        # mesh.export(mesh_path)
        return mesh
    except Exception as e:
        print(f"Error executing CadQuery code : {e}")
        return None