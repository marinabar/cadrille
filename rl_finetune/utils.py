import concurrent
import multiprocessing
import os
import random

from collections import defaultdict
from argparse import ArgumentParser
from multiprocessing import Process
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

os.environ["PYGLET_HEADLESS"] = "True"

import trimesh
from scipy.spatial import cKDTree

from torch.utils.data import DataLoader
from tqdm import tqdm


code_prefix = """
import cadquery as cq
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def compute_iou(pred_mesh, gt_mesh):
    intersection_volume = 0
    for gt_mesh_i in gt_mesh.split():
        for pred_mesh_i in pred_mesh.split():
            intersection = gt_mesh_i.intersection(pred_mesh_i)
            volume = intersection.volume if intersection is not None else 0
            intersection_volume += volume

    gt_volume = sum(m.volume for m in gt_mesh.split())
    pred_volume = sum(m.volume for m in pred_mesh.split())
    union_volume = gt_volume + pred_volume - intersection_volume
    iou = intersection_volume / (union_volume + 1e-6)
    return iou

def compute_cd(pred_mesh, gt_mesh):
    n_points = 8192
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    cd = np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))
    return cd

def transform_real_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
    mesh.apply_scale(2.0 / max(mesh.extents))  # Normalize to [-1, 1]
    return mesh

def tessellate(compound):
    # compound = cq.importers.importStep(tmp_file_name).val()
    import trimesh
    vertices, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
    return mesh

def transform_gt_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
    mesh.apply_scale(1.0 / max(mesh.extents))  # Normalize to [0, 1]
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

"""

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


def run_code(code, queue):
    """Executes code and puts results in a queue."""
    local_vars = {}
    try:
        exec(code, {}, local_vars)  # Run the code
        exec_result = local_vars.get("iou", None)
        queue.put(exec_result)  # Put result in queue
    except Exception as e:
        # queue.put(f"Error: {str(e)}")  # Send error message
        queue.put(None)  # Send error message


def run_code_with_timeout(code, timeout=5):
    """Runs code with a timeout and retrieves results."""
    queue = multiprocessing.SimpleQueue()
    process = multiprocessing.Process(target=run_code, args=(code, queue))

    process.start()
    process.join(timeout)

    if process.is_alive():
        print("Timout while running code", flush=True)
        process.terminate()  # Kill the process
        process.join()

        raise RuntimeError("Execution timed out")

    return queue.get()  # Get the result


def get_metrics_from_single_text(text, gt_file, pred_mesh_path, pred_brep_path, n_points):

    base_file = os.path.basename(gt_file).rsplit('.stl', 1)[0]

    mesh_path = os.path.join(pred_mesh_path, os.path.basename(gt_file))
    brep_path = os.path.join(pred_brep_path, base_file + '.step')

    # runs the python code and saves the mesh and brep files
    code_to_mesh_and_brep_safe(text, mesh_path, brep_path)
    
    cd, iou = None, None
    try:  # apply_transform fails for some reason; or mesh path can not exist
        pred_mesh = trimesh.load_mesh(mesh_path)
        # do not normalize predicted mesh
        #pred_mesh = transform_pred_mesh(pred_mesh)
        
        gt_mesh = trimesh.load_mesh(gt_file)
        gt_mesh = transform_gt_mesh(gt_mesh)
        cd = compute_cd(gt_mesh, pred_mesh, n_points)
        iou = compute_iou(gt_mesh, pred_mesh)
    except:
        pass
    
    return dict(file_name=base_file, cd=cd, iou=iou)


def get_metrics_from_texts(texts, meshes, max_workers=None):
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
    pred_mesh_path = os.path.join(os.path.dirname(temp_path), 'tmp_mesh')
    pred_brep_path = os.path.join(os.path.dirname(temp_path), 'tmp_brep')
    n_points = 8192
    results = []
    if max_workers is None:
        max_workers = os.cpu_count()

    
    with NonDaemonPool(max_workers) as pool:
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
            pred_ious = get_metrics_from_texts(decoded_texts, batch["mesh_path"])
            for i, pred_iou in enumerate(pred_ious):
                if pred_iou is None:
                    n_incorrect += 1
                    continue
                if pred_iou < 0:
                    n_failed_intersect += 1
                else:
                    ious.append(pred_iou)

    print(f"IoU mean {np.mean(ious)}, median {np.median(ious)}")
    print(f"CD mean {np.mean(cds)}, median {np.median(cds)}")
    print(f"Invalid generations fraction: {n_incorrect / len(eval_examples)}")
    print(f"Intersect failure fraction: {n_failed_intersect / len(eval_examples)}")
    print("=" * 50)

    model.train()
    return ious, cds, n_incorrect / len(eval_examples), n_failed_intersect / len(eval_examples)


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
    # saves mesh and brep files from code string 
    try:
        exec(code_str, globals())
        compound = globals()['r'].val()
        mesh = compound_to_mesh(compound)
        assert len(mesh.faces) > 2
        mesh.export(mesh_path)
        # cq.exporters.export(compound, brep_path)
    except:
        pass


def code_to_mesh_and_brep_safe(code_str, mesh_path, brep_path):
    p = Process(target=code_to_mesh_and_brep, args=(code_str, mesh_path, brep_path))
    p.start()
    p.join(3)
    if p.is_alive():
        p.terminate()
        p.join()