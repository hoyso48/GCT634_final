import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
import librosa
import pyloudnorm as pyln
from tqdm import tqdm

from kadtk.model_loader import ModelLoader
from kadtk.utils import PathOrPaths, PathLike, _get_files

# Functions from original clap_score.py
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

class ClapScore:
    def __init__(self, ml: ModelLoader, device: str, logger=None, **kwargs):
        self.ml = ml
        self.device = torch.device(device)
        self.logger = logger
        torch.autograd.set_grad_enabled(False)
        
        # Load the model
        if self.ml.model is None:
            self.ml.load_model()
        self.model = self.ml.model
        self.model.to(self.device)
        self.model.eval()

    def score(self, audio_source: PathOrPaths, text_source: Union[Path, list[str]]):
        if isinstance(text_source, (str, Path)):
            captions_df = pd.read_csv(text_source)
            if not all(col in captions_df.columns for col in ['filename', 'caption']):
                raise ValueError("CSV file for captions must contain 'filename' and 'caption' columns.")
            
            audio_files = _get_files(audio_source)
            audio_map = {f.name: f for f in audio_files}

            file_list = []
            text_list = []
            for _, row in captions_df.iterrows():
                if row['filename'] in audio_map:
                    file_list.append(audio_map[row['filename']])
                    text_list.append(row['caption'])
            
            if len(file_list) != len(captions_df):
                self.logger.warning(f"Found {len(file_list)} matching audio files out of {len(captions_df)} captions.")

        elif isinstance(text_source, list):
            file_list = _get_files(audio_source)
            text_list = text_source
            if len(file_list) != len(text_list):
                raise ValueError("The number of audio files and text captions must be the same.")
        else:
            raise TypeError("`text_source` must be a path to a CSV file or a list of strings.")

        if not file_list:
            self.logger.error("No audio files found to score.")
            return 0.0

        return np.mean(self.score_individual(file_list, text_list))
        
    def score_individual(self, audio_source: PathOrPaths, text_source: Union[Path, list[str]]):
        if isinstance(text_source, (str, Path)):
            captions_df = pd.read_csv(text_source)
            if not all(col in captions_df.columns for col in ['filename', 'caption']):
                raise ValueError("CSV file for captions must contain 'filename' and 'caption' columns.")
            
            audio_files = _get_files(audio_source)
            audio_map = {f.name: f for f in audio_files}

            file_list = []
            text_list = []
            for _, row in captions_df.iterrows():
                if row['filename'] in audio_map:
                    file_list.append(audio_map[row['filename']])
                    text_list.append(row['caption'])
            
            if len(file_list) != len(captions_df):
                self.logger.warning(f"Found {len(file_list)} matching audio files out of {len(captions_df)} captions.")

        elif isinstance(text_source, list):
            file_list = _get_files(audio_source)
            text_list = text_source
            if len(file_list) != len(text_list):
                raise ValueError("The number of audio files and text captions must be the same.")
        else:
            raise TypeError("`text_source` must be a path to a CSV file or a list of strings.")

        if not file_list:
            self.logger.error("No audio files found to score.")
            return []

        text_embeddings = self._get_text_embeddings(text_list)
        audio_embeddings = self._get_audio_embeddings(file_list)
        
        # Calculate cosine similarity
        audio_embeddings = audio_embeddings.to(self.device)
        text_embeddings = text_embeddings.to(self.device)

        sim = torch.nn.functional.cosine_similarity(audio_embeddings, text_embeddings, dim=1, eps=1e-8)
        
        return sim.cpu().tolist()

    def _get_text_embeddings(self, text_list, batch_size=32):
        all_embs = []
        for i in tqdm(range(0, len(text_list), batch_size), desc="Computing text embeddings"):
            batch = text_list[i:i+batch_size]
            embs = self.model.get_text_embedding(batch, use_tensor=True)
            all_embs.append(embs)
        return torch.cat(all_embs)

    def _get_audio_embeddings(self, audio_path_list, batch_size=32):
        all_embs = []
        for i in tqdm(range(0, len(audio_path_list), batch_size), desc="Computing audio embeddings"):
            batch_paths = audio_path_list[i:i+batch_size]
            audio_batch = []
            for f_path in batch_paths:
                audio, _ = librosa.load(f_path, sr=48000, mono=True)
                audio = pyln.normalize.peak(audio, -1.0)
                audio = torch.from_numpy(int16_to_float32(float32_to_int16(audio))).float()
                audio_batch.append(audio)
            
            embs = self.model.get_audio_embedding_from_data(x=audio_batch, use_tensor=True)
            all_embs.append(embs)
        return torch.cat(all_embs) 