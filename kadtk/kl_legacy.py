import torch
from pathlib import Path
import numpy as np
from typing import Union
import traceback
from hypy_utils import write
from hypy_utils.tqdm_utils import tmap, tq

from kadtk.emb_loader import EmbeddingLoader
from kadtk.model_loader import ModelLoader
from kadtk.utils import PathOrPaths, PathLike, _get_files

def calculate_kl_divergence(
    features_1: np.ndarray,
    features_2: np.ndarray,
    device: str
):
    """
    Calculate KL-divergence between two sets of features.
    features_1: predicted features (numpy array)
    features_2: ground truth features (numpy array)
    """
    EPS = 1e-6
    features_1 = torch.from_numpy(features_1).to(device)
    features_2 = torch.from_numpy(features_2).to(device)

    # Softmax KL
    kl_softmax = torch.nn.functional.kl_div(
        (features_1.softmax(dim=1) + EPS).log(),
        features_2.softmax(dim=1),
        reduction="sum",
    ) / len(features_1)

    kl_ref = torch.nn.functional.kl_div(
        (features_1.softmax(dim=1) + EPS).log(),
        features_2.softmax(dim=1),
        reduction="none",
    ) / len(features_1)
    kl_ref = torch.mean(kl_ref, dim=-1)

    # Sigmoid KL
    kl_sigmoid = torch.nn.functional.kl_div(
        (features_1.sigmoid() + EPS).log(),
        features_2.sigmoid(),
        reduction="sum",
    ) / len(features_1)

    return {
        "kullback_leibler_divergence_sigmoid": kl_sigmoid.item(),
        "kullback_leibler_divergence_softmax": kl_softmax.item(),
        "kullback_leibler_divergence_ref": kl_ref.item(),
    }

class KullbackLeiblerDivergence:
    def __init__(self, ml: ModelLoader, device: str, audio_load_worker: int = 8, logger=None, **kwargs):
        self.ml = ml
        self.device = torch.device(device)
        self.emb_loader = EmbeddingLoader(ml, load_model=False)
        self.audio_load_worker = audio_load_worker
        self.logger = logger
        self.baseline_dir_for_cache = None
        self.eval_dir_for_cache = None
        torch.autograd.set_grad_enabled(False)

    def score(self, baseline: PathOrPaths, eval: PathOrPaths):
        baseline_files = {p.name: p for p in _get_files(baseline)}
        eval_files = {p.name: p for p in _get_files(eval)}
        
        common_files = sorted(list(baseline_files.keys() & eval_files.keys()))

        if not common_files:
            self.logger.error("No common audio files found between baseline and eval directories.")
            return -1.0
        
        self.logger.info(f"Found {len(common_files)} common files.")
        
        baseline_embs = []
        eval_embs = []

        for f_name in tq(common_files, desc="Loading embeddings"):
            eval_emb = self.emb_loader.read_embedding_file(eval_files[f_name])
            baseline_emb = self.emb_loader.read_embedding_file(baseline_files[f_name])

            eval_embs.append(np.mean(eval_emb, axis=0))
            baseline_embs.append(np.mean(baseline_emb, axis=0))

        features_1 = np.array(eval_embs)
        features_2 = np.array(baseline_embs)
        
        kl_scores = calculate_kl_divergence(features_1, features_2, self.device)

        self.logger.info(f"KL Divergence (softmax): {kl_scores['kullback_leibler_divergence_softmax']:.4f}")
        self.logger.info(f"KL Divergence (sigmoid): {kl_scores['kullback_leibler_divergence_sigmoid']:.4f}")
        self.logger.info(f"KL Divergence (ref): {kl_scores['kullback_leibler_divergence_ref']:.4f}")
        return kl_scores['kullback_leibler_divergence_ref']

    def score_individual(self, baseline: PathOrPaths, eval_dir: PathOrPaths, csv_name: Union[Path, str, None]) -> Path:
        baseline_files = {p.name: p for p in _get_files(baseline)}
        eval_files = {p.name: p for p in _get_files(eval_dir)}
        
        common_files = sorted(list(baseline_files.keys() & eval_files.keys()))

        if csv_name is None:
            csv = Path('data') / 'result' / self.ml.name / "kl-indiv.csv"
        else:
            csv = Path(csv_name)
        
        if csv.exists():
            self.logger.info(f"CSV file {csv} already exists, skipping...")
            return csv

        if not common_files:
            self.logger.error("No common audio files found between baseline and eval directories.")
            csv.parent.mkdir(parents=True, exist_ok=True)
            csv.touch() # create empty file
            return csv

        self.logger.info(f"Found {len(common_files)} common files for individual KL calculation.")

        def _kl_helper(f_name):
            try:
                eval_emb = self.emb_loader.read_embedding_file(eval_files[f_name])
                baseline_emb = self.emb_loader.read_embedding_file(baseline_files[f_name])

                eval_emb_mean = np.mean(eval_emb, axis=0, keepdims=True)
                baseline_emb_mean = np.mean(baseline_emb, axis=0, keepdims=True)
                
                kl_scores = calculate_kl_divergence(eval_emb_mean, baseline_emb_mean, self.device)
                kl_softmax = kl_scores['kullback_leibler_divergence_softmax']
                return f_name, kl_softmax

            except Exception as e:
                traceback.print_exc()
                self.logger.error(f"An error occurred calculating individual KL for {f_name}: {e}")
                return f_name, None

        results = tmap(_kl_helper, common_files, desc="Calculating individual KL scores", max_workers=self.audio_load_worker)

        pairs = [p for p in results if p[1] is not None]
        pairs = sorted(pairs, key=lambda x: np.abs(x[1]))
        
        csv.parent.mkdir(parents=True, exist_ok=True)
        write(csv, "file,kl_softmax\n" + "\n".join([f"{p[0]},{p[1]}" for p in pairs]))
        self.logger.info(f"Individual KL scores saved to {csv}")

        return csv 