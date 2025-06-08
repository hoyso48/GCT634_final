import multiprocessing
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchaudio
from hypy_utils.tqdm_utils import tmap, tq, pmap

from kadtk.model_loader import ModelLoader
from kadtk.utils import find_sox_formats, get_cache_embedding_path, PathOrPaths, _get_files

sox_path = os.environ.get('SOX_PATH', 'sox')
ffmpeg_path = os.environ.get('FFMPEG_PATH', 'ffmpeg')
TORCHAUDIO_RESAMPLING = True

if not(TORCHAUDIO_RESAMPLING):
    if not shutil.which(sox_path):
        raise Exception(f"Could not find SoX executable at {sox_path}, please install SoX and set the SOX_PATH environment variable.")
    if not shutil.which(ffmpeg_path):
        raise Exception(f"Could not find ffmpeg executable at {ffmpeg_path}, please install ffmpeg and set the FFMPEG_PATH environment variable.")

class EmbeddingLoader:
    def __init__(self, model: ModelLoader, audio_load_worker: int = 8, load_model: bool = True):
        self.ml = model
        self.audio_load_worker = audio_load_worker
        self.sox_formats = find_sox_formats(sox_path)
        if load_model:
            self.ml.load_model()
            self.loaded = True

    def load_audio(self, f: Union[str, Path]):
        f = Path(f)

        # Use a temporary file for on-the-fly resampling without permanent caching
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
            resampled_path = Path(tmp_f.name)
        
        try:
            if TORCHAUDIO_RESAMPLING:
                x, fsorig = torchaudio.load(f)
                
                # Resample if the sample rate is different
                if fsorig != self.ml.sr:
                    if x.shape[0] > 1:
                        # Use torchaudio's remix for safer mono conversion
                        x = torch.mean(x, 0, keepdim=True)
                    
                    resampler = torchaudio.transforms.Resample(
                        fsorig,
                        self.ml.sr,
                        lowpass_filter_width=64,
                        rolloff=0.9475937167399596,
                        resampling_method="sinc_interp_kaiser",
                        beta=14.769656459379492,
                    )
                    resampled_x = resampler(x)
                else:                
                    resampled_x = x

                torchaudio.save(resampled_path, resampled_x, self.ml.sr, encoding="PCM_S", bits_per_sample=16)

            else: # SOX logic for resampling
                sox_args = ['-r', str(self.ml.sr), '-c', '1', '-b', '16']
    
                if f.suffix[1:] not in self.sox_formats:
                    # Use ffmpeg for format conversion and then pipe to sox for resampling
                    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_format_conv:
                        subprocess.run([ffmpeg_path, "-hide_banner", "-loglevel", "error", "-i", f, tmp_format_conv.name], check=True)
                        subprocess.run([sox_path, tmp_format_conv.name, *sox_args, resampled_path], check=True)
                else:
                    # Use sox directly for resampling
                    subprocess.run([sox_path, f, *sox_args, resampled_path], check=True)
            
            # Load the resampled audio data using the model-specific loader
            wav_data = self.ml.load_wav(resampled_path)

        finally:
            # Ensure the temporary file is always cleaned up
            if os.path.exists(resampled_path):
                os.remove(resampled_path)
                
        return wav_data
    
    def read_embedding_file(self, audio_dir: Union[str, Path]):
        """
        Read embedding from a cached file.
        """
        cache = get_cache_embedding_path(self.ml.name, audio_dir)
        if not cache.exists():
            raise ValueError(f"Embedding file {cache} does not exist.")
        emb = np.load(cache)
        return emb

    def load_embeddings(self, dir_or_files: PathOrPaths) -> np.ndarray:
        """
        Load embeddings from a directory or a list of files.
        If the embeddings are cached, load from cache. Otherwise, calculate them and cache.
        """
        files = _get_files(dir_or_files)
        
        # If all of the embedding files are cached, load them directly
        if all([get_cache_embedding_path(self.ml.name, f).exists() for f in files]):
            embeds = [np.load(get_cache_embedding_path(self.ml.name, f)) for f in files]
            return np.concatenate(embeds, axis=0)

        # Otherwise, load the audio files and calculate the embeddings
        return self._load_embeddings(files, concat=True)

    def _load_embeddings(self, files: list[Path], concat: bool = False):
        """
        Helper function to load embeddings from a list of audio files.
        """
        if self.ml.model is None:
            self.ml.load_model()

        # Load audio files sequentially instead of in parallel
        audios = [self.load_audio(f) for f in tq(files, desc=f"Loading audio files for {self.ml.name}")]

        # Get embeddings
        if concat:
            # For FAD, we need to concatenate all embeddings
            embs = np.concatenate([self.ml.get_embedding(a) for a in tq(audios, desc=f"Calculating embeddings for {self.ml.name}")], axis=0)
        else:
            # For KAD, we need to keep the embeddings separate
            embs = [self.ml.get_embedding(a) for a in tq(audios, desc=f"Calculating embeddings for {self.ml.name}")]
        return embs

    def cache_embedding_file(self, audio_dir: Union[str, Path]):
        """
        Compute embedding for an audio file and cache it to a file.
        """
        cache = get_cache_embedding_path(self.ml.name, audio_dir)

        if cache.exists():
            return

        if self.ml.model is None:
            self.ml.load_model()

        # Load file, get embedding, save embedding
        wav_data = self.load_audio(audio_dir)
        embd = self.ml.get_embedding(wav_data)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, embd)

    def get_or_compute_embedding(self, audio_path: Path, use_cache: bool):
        """
        Get embedding for an audio file, using cache if available and enabled.
        If computed, it will be cached only if use_cache is True.
        """
        if use_cache:
            cache = get_cache_embedding_path(self.ml.name, audio_path)
            if cache.exists():
                return np.load(cache)

        # If not cached or caching disabled, compute it
        if self.ml.model is None:
            self.ml.load_model()

        wav_data = self.load_audio(audio_path)
        embd = self.ml.get_embedding(wav_data)

        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, embd)
        
        return embd

# Main
def _cache_embedding_batch(args):
    fs: list[Path]
    ml: ModelLoader
    fs, ml, kwargs = args
    emb_loader = EmbeddingLoader(ml, **kwargs)
    for f in fs:
        print(f"Loading {f} using {ml.name}")
        emb_loader.cache_embedding_file(f)


def cache_embedding_files(files: Union[list[Path], str, Path], ml: ModelLoader, workers: int = 8, 
                          force_emb_encode: bool = False, **kwargs):
    """
    Get embeddings for all audio files in a directory.

    Params:
    - files (list[Path] | str | Path): List of audio files or a directory containing audio files.
    - ml (ModelLoader): ModelLoader instance to use.
    - workers (int): Number of workers to use.
    - force_emb_encode (bool): If True, re-extract embeddings even if they already exist.
    """
    if isinstance(files, (str, Path)):
        files = list(Path(files).glob('*.*'))

    if force_emb_encode:
        emb_path = files[0].parent / "embeddings" / ml.name
        if os.path.exists(emb_path):
            # Remove the folder and its contents
            shutil.rmtree(emb_path)
            print(f"The folder '{emb_path}' has been successfully removed.")
        

    # Filter out files that already have embeddings
    files = [f for f in files if not get_cache_embedding_path(ml.name, f).exists()]

    if len(files) == 0:
        print("All files already have embeddings, skipping.")
        return

    print(f"Loading {len(files)} audio files...")

    # Split files into batches
    batches = list(np.array_split(files, workers))
    
    # Cache embeddings in parallel
    multiprocessing.set_start_method('spawn', force=True)
    with torch.multiprocessing.Pool(workers) as pool:
        pool.map(_cache_embedding_batch, [(b, ml, kwargs) for b in batches])
        