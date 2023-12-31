from marius.tools.preprocess.datasets.fb15k_237 import FB15K237
import executor as e
import reporting
from pathlib import Path

BASE_PATH = Path("experiment_manager/setup_dgl/configs")


def run_setup_dgl(dataset_dir, results_dir, overwrite, enable_dstat, enable_nvidia_smi, show_output, short, num_runs=5):
    """
    Models: DistMult, GraphSage, GAT
    Systems: Marius, DGL, PyG
    """

    dataset_name = "fb15k237"

    dgl_dm_config = BASE_PATH / Path("dgl_dm.txt")

    if not (dataset_dir / Path(dataset_name) / Path("edges/train_edges.bin")).exists():
        print("==== Preprocessing {} =====".format(dataset_name))
        dataset = FB15K237(dataset_dir / Path(dataset_name))
        dataset.download()
        dataset.preprocess()
    else:
        print("==== {} already preprocessed =====".format(dataset_name))

    for i in range(1):

        # Run DGL
        e.run_config(dgl_dm_config, results_dir / Path("DGL_SETUP"), overwrite, enable_dstat, enable_nvidia_smi,
                     show_output, i, "dgl")
