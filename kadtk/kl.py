import torch
from pathlib import Path
import numpy as np
import logging
from typing import Union

from hypy_utils.tqdm_utils import tq

from kadtk.emb_loader import EmbeddingLoader
from kadtk.model_loader import ModelLoader
from kadtk.utils import PathOrPaths

# Configure logger
log = logging.getLogger(__name__)

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
    print("features_1.shape, features_2.shape", features_1.shape, features_2.shape)
    kl_softmax = torch.nn.functional.kl_div(
        (features_1.softmax(dim=1) + EPS).log(),
        features_2.softmax(dim=1),
        reduction="sum",
    ) / len(features_1)

    # Sigmoid KL
    kl_sigmoid = torch.nn.functional.kl_div(
        (features_1.sigmoid() + EPS).log(),
        features_2.sigmoid(),
        reduction="sum",
    ) / len(features_1)

    return {
        "kullback_leibler_divergence_sigmoid": kl_sigmoid.item(),
        "kullback_leibler_divergence_softmax": kl_softmax.item(),
    }

class KL:
    def __init__(self, ml: ModelLoader, device: str, audio_load_worker: int = 8, logger=None, **kwargs):
        self.ml = ml
        self.device = torch.device(device)
        self.emb_loader = EmbeddingLoader(ml, load_model=False)
        self.audio_load_worker = audio_load_worker
        self.logger = logger if logger else log
        self.baseline_dir_for_cache = None
        self.eval_dir_for_cache = None
        torch.autograd.set_grad_enabled(False)

    def score(self, baseline_files: list[Path], eval_files: list[Path]):
        if len(baseline_files) != len(eval_files):
            raise ValueError("Baseline and eval file lists must have the same length for KL divergence calculation.")

        if not baseline_files:
            self.logger.error("Input file lists are empty.")
            return -1.0
        
        self.logger.info(f"Calculating KL for {len(baseline_files)} paired files.")
        
        baseline_embs = []
        eval_embs = []

        use_cache_baseline = self.baseline_dir_for_cache is not None
        use_cache_eval = self.eval_dir_for_cache is not None

        for i in tq(range(len(baseline_files)), desc="Loading embeddings for KL"):
            baseline_file = baseline_files[i]
            eval_file = eval_files[i]

            baseline_emb = self.emb_loader.get_or_compute_embedding(baseline_file, use_cache=use_cache_baseline)
            eval_emb = self.emb_loader.get_or_compute_embedding(eval_file, use_cache=use_cache_eval)
            
            # Take the mean of embeddings for each file
            baseline_embs.append(np.mean(baseline_emb, axis=0))
            eval_embs.append(np.mean(eval_emb, axis=0))

        features_1 = np.array(eval_embs)
        features_2 = np.array(baseline_embs)
        
        kl_scores = calculate_kl_divergence(features_1, features_2, self.device)

        self.logger.info(f"KL Divergence (softmax): {kl_scores['kullback_leibler_divergence_softmax']:.4f}")
        self.logger.info(f"KL Divergence (sigmoid): {kl_scores['kullback_leibler_divergence_sigmoid']:.4f}")
        return kl_scores['kullback_leibler_divergence_softmax']