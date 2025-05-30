import os
import trimesh
import numpy as np
import cadquery as cq
from tqdm import tqdm
from functools import partial
from scipy.spatial import cKDTree
from collections import defaultdict
from argparse import ArgumentParser
from multiprocessing import Process
from multiprocessing.pool import Pool


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


def compute_chamfer_distance(gt_mesh, pred_mesh, n_points):
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    return np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))


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


def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def py_file_to_mesh_and_brep_files(py_path, mesh_path, brep_path):
    try:
        with open(py_path, 'r') as f:
            py_string = f.read()
        exec(py_string, globals())
        compound = globals()['r'].val()
        mesh = compound_to_mesh(compound)
        assert len(mesh.faces) > 2
        mesh.export(mesh_path)
        # comment this line if no need to export brep
        # cq.exporters.export(compound, brep_path)
    except:
        pass


def py_file_to_mesh_and_brep_files_safe(py_path, mesh_path, brep_path):
    process = Process(
        target=py_file_to_mesh_and_brep_files,
        args=(py_path, mesh_path, brep_path))
    process.start()
    process.join(3)

    if process.is_alive():
        print('process alive:', py_path)
        process.terminate()
        process.join()


def run_cd_single(py_file_name, pred_py_path, pred_mesh_path, pred_brep_path, gt_mesh_path, n_points, normalize_std):
    eval_file_name = py_file_name[:py_file_name.rfind('+')]
    py_path = os.path.join(pred_py_path, py_file_name)
    mesh_path = os.path.join(pred_mesh_path, py_file_name[:-3] + '.stl')
    brep_path = os.path.join(pred_brep_path, py_file_name[:-3] + '.step')
    py_file_to_mesh_and_brep_files_safe(py_path, mesh_path, brep_path)
    
    cd, iou = None, None
    try:  # apply_transform fails for some reason; or mesh path can not exist
        pred_mesh = trimesh.load_mesh(mesh_path)
        pred_mesh.apply_transform(trimesh.transformations.scale_matrix(1 / normalize_std / 2))
        pred_mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
        gt_mesh = trimesh.load_mesh(os.path.join(gt_mesh_path, eval_file_name + '.stl'))
        cd = compute_chamfer_distance(gt_mesh, pred_mesh, n_points)
        iou = compute_iou(gt_mesh, pred_mesh)
    except:
        pass
    
    index = py_file_name[len(eval_file_name) + 1: -3]
    return dict(file_name=eval_file_name, id=index, cd=cd, iou=iou)


def run(gt_mesh_path, pred_py_path, n_points, normalize_std):
    pred_mesh_path = os.path.join(os.path.dirname(pred_py_path), 'tmp_mesh')
    pred_brep_path = os.path.join(os.path.dirname(pred_py_path), 'tmp_brep')
    best_names_path = os.path.join(os.path.dirname(pred_py_path), 'tmp.txt')

    # should be no predicted meshes from previous experiments
    os.makedirs(pred_mesh_path, exist_ok=True)
    os.makedirs(pred_brep_path, exist_ok=True)
    assert len(os.listdir(pred_mesh_path)) == len(os.listdir(pred_brep_path)) == 0

    # compute chamfer distance and iou for each sample
    py_file_names = os.listdir(pred_py_path)
    with NonDaemonPool(16) as pool:
        py_metrics = list(tqdm(pool.imap(
            partial(
                run_cd_single,
                pred_py_path=pred_py_path,
                pred_mesh_path=pred_mesh_path,
                pred_brep_path=pred_brep_path,
                gt_mesh_path=gt_mesh_path,
                n_points=n_points,
                normalize_std=normalize_std),
            py_file_names), total=len(py_file_names)))

    # aggregate metrics per eval_file_name
    metrics = defaultdict(lambda: defaultdict(list))
    for m in py_metrics:
        if m['cd'] is not None:
            metrics[m['file_name']]['cd'].append(m['cd'])
            metrics[m['file_name']]['id'].append(m['id'])
        if m['iou'] is not None:
            metrics[m['file_name']]['iou'].append(m['iou'])

        # empty value for invalid predictions
        metrics[m['file_name']]

    
    # select best metrics per eval_file_name
    ir_cd, ir_iou, cd, iou, best_names = 0, 0, list(), list(), list()
    for key, value in metrics.items():
        if len(value['cd']):
            argmin = np.argmin(value['cd'])
            cd.append(value['cd'][argmin])
            index = value['id'][argmin]
            best_names.append(f'{key}+{index}.py')
        else:
            ir_cd += 1
        
        if len(value['iou']):
            iou.append(np.max(value['iou']))
        else:
            ir_iou += 1

    with open(best_names_path, 'w') as f:
        f.writelines([line + '\n' for line in best_names])

    print(f'mean iou: {np.mean(iou):.3f}',
          f'median cd: {np.median(cd) * 1000:.3f}')

    cd = sorted(cd)
    for i in range(5):
        print(f'skip: {i} ir: {(ir_cd + i) / len(metrics) * 100:.2f}',
              f'mean cd: {np.mean(cd[:len(cd) - i]) * 1000:.3f}')


# To overcome CadQuery memory leaks, we call each exec() in a separate Process with
# timeout of 3 seconds. The Pool is tweaked to support non-daemon processes that can
# call one more nested process.
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gt-mesh-path', type=str, default='./data/deepcad_test_mesh')
    parser.add_argument('--pred-py-path', type=str, default='./work_dirs/tmp_py')
    parser.add_argument('--n-points', type=int, default=8192)
    parser.add_argument('--normalize_std', type=float, default=100)
    args = parser.parse_args()
    run(
        args.gt_mesh_path, args.pred_py_path, args.n_points, args.normalize_std)
