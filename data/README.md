
Ensure *git-lsf* is installed and *git clone* the datasets from our :hugs: HuggingFace.


- DeepCAD [test](https://huggingface.co/datasets/maksimko123/deepcad_test_mesh). Meshes are produced by official DeepCAD [script](https://github.com/ChrisWu1997/DeepCAD/blob/master/dataset/json2pc.py) and normalized to the unit cube.
- Fusion360 [test](https://huggingface.co/datasets/maksimko123/fusion360_test_mesh). Meshes are downloaded from [link](https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/docs/reconstruction.md#traintest-split) and normalized to unit cube.
- Text2CAD [train / val / test](https://huggingface.co/datasets/maksimko123/text2cad). Text prompts are downloaded from [link](https://github.com/SadilKhan/Text2CAD?tab=readme-ov-file#-data-preparation) and shortened a bit. We also provide CadQuery codes for almost all DeepCAD examples.
- CAD-Recode [train / val](https://huggingface.co/datasets/filapro/cad-recode-v1.5). To convert CadQuery programs to meshes before training run *cadrecode2mesh.py* script.

Overall data structure should be as follows:
```
data
└── cad-recode-v1.5
    ├── train
        ├── batch_00
            ├── 0.py
            ├── 0.stl
            └── ...
        └── ...
    ├── val
        ├── 0.py
        ├── 0.stl
        └── ...
    ├── train.pkl
    └── val.pkl
    ├── text2cad
        ├── cadquery
            ├── 0.py
            └── ...
        ├── train.pkl
        ├── val.pkl
        └── test.pkl
    ├── deepcad_test_mesh
        ├── 0.stl
        └── ...
    └── fusion360_test_mesh
        ├── 0.stl
        └── ...
```