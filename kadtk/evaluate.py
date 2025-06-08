import argparse
import pandas as pd
import os
import torch
import gc
from tqdm import tqdm
from pathlib import Path
import librosa
import numpy as np
import pyloudnorm as pyln

from kadtk.model_loader import get_all_models, ModelLoader
from kadtk.fad import FrechetAudioDistance
from kadtk.kad import KernelAudioDistance
from kadtk.kl import KL
from kadtk.clap import ClapScore

def run_clap_evaluation(model_loader: ModelLoader, captions: list[str], audio_paths: list[str], device: str):
    clap_scorer = ClapScore(ml=model_loader, device=device)
    return clap_scorer.score_individual(audio_paths, captions)

def run_audiobox_evaluation(audio_paths: list[str], batch_size: int):
    try:
        from audiobox_aesthetics.infer import initialize_predictor
    except ImportError:
        print("Audiobox aesthetics model not found. Please install it to use this metric.")
        return None

    print("Initializing Audiobox predictor...")
    audiobox_model = initialize_predictor()
    
    scores = []
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Evaluating with Audiobox"):
        batch_paths = audio_paths[i:i+batch_size]
        paths_dict = [{"path": p} for p in batch_paths]
        scores.extend(audiobox_model.forward(paths_dict))
    
    return scores

def run_distributional_evaluation(metric_name: str, model_loader: ModelLoader, baseline_paths: list[str], eval_paths: list[str], device: str, baseline_dir_for_cache: str, eval_dir_for_cache: str):
    METRIC_MAP = {
        'fad': FrechetAudioDistance,
        'kad': KernelAudioDistance,
        'kl': KL,
    }
    
    metric_class = METRIC_MAP[metric_name]
    metric = metric_class(model_loader, device=device)
    metric.baseline_dir_for_cache = baseline_dir_for_cache
    metric.eval_dir_for_cache = eval_dir_for_cache
    
    score = metric.score(baseline_paths, eval_paths)
    
    if isinstance(score, torch.Tensor):
        return score.item()
    return score

def main():
    models = {m.name: m for m in get_all_models()}
    parser = argparse.ArgumentParser(description="Comprehensive evaluation script for audio generation models.")
    
    # Paths
    parser.add_argument("--eval_csv_path", type=str, required=True, help="CSV with 'id' and 'wav_path' for evaluation files.")
    parser.add_argument("--eval_wav_dir", type=str, required=True, help="Directory with evaluation audio files.")
    parser.add_argument("--output_csv_path", type=str, required=True, help="Path to save the output scores CSV.")
    parser.add_argument("--caption_csv_path", type=str, help="CSV with 'id' and 'caption', required for CLAP score.")
    parser.add_argument("--baseline_csv_path", type=str, help="CSV for baseline files, required for FAD/KAD/KL.")
    parser.add_argument("--baseline_wav_dir", type=str, help="Directory with baseline audio files.")

    # Metrics and Models
    parser.add_argument("--metrics", type=str, nargs='+', required=True, choices=['clap', 'audiobox', 'fad', 'kad', 'kl'], help="List of metrics to compute.")
    parser.add_argument("--clap_model", type=str, default="clap-laion-music", choices=[m for m in models if 'clap' in m], help="CLAP model to use.")
    parser.add_argument("--kadtk_model", type=str, help="KADtk model for FAD/KAD/KL.", choices=list(models.keys()))

    # Config
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache_eval_stats", action='store_true', help="If set, caches the statistics for the evaluation set. Off by default.")

    args = parser.parse_args()

    # --- Argument Validation ---
    dist_metrics_req = any(m in args.metrics for m in ['fad', 'kad', 'kl'])
    if dist_metrics_req and (not args.baseline_csv_path or not args.baseline_wav_dir or not args.kadtk_model):
        raise ValueError("--baseline_csv_path, --baseline_wav_dir, and --kadtk_model are required for FAD, KAD, or KL.")
    if 'clap' in args.metrics and not args.caption_csv_path:
        raise ValueError("--caption_csv_path is required for CLAP score.")

    # --- Data Loading ---
    df_eval = pd.read_csv(args.eval_csv_path)
    required_columns = ['id', 'wav_path']
    if not all(col in df_eval.columns for col in required_columns):
        raise ValueError(f"Eval CSV must contain {required_columns} columns.")
    df_eval = df_eval[required_columns]

    if 'clap' in args.metrics:
        df_captions = pd.read_csv(args.caption_csv_path)
        if not all(col in df_captions.columns for col in ['id', 'caption']):
            raise ValueError("Caption CSV must contain 'id' and 'caption' columns.")
        df_captions = df_captions[['id', 'caption']]
        df_eval = pd.merge(df_eval, df_captions, on='id', how='inner')

    eval_wav_paths_all = [os.path.join(args.eval_wav_dir, p) for p in df_eval['wav_path'].tolist()]
    df_results = df_eval.copy()

    # --- Metric Calculation ---
    if 'clap' in args.metrics:
        print("--- Calculating CLAP Scores ---")
        clap_model_loader = models[args.clap_model]
        
        # Regular CLAP score
        clap_scores = run_clap_evaluation(clap_model_loader, df_results.caption.tolist(), eval_wav_paths_all, args.device)
        df_results['clap_score'] = clap_scores
        
        # Synthesized CLAP score
        synth_captions = ['Synthesized ' + c for c in df_results.caption.tolist()]
        clap_scores_synth = run_clap_evaluation(clap_model_loader, synth_captions, eval_wav_paths_all, args.device)
        df_results['clap_score_synthesized'] = clap_scores_synth

        del clap_model_loader
        gc.collect()
        torch.cuda.empty_cache()

    if 'audiobox' in args.metrics:
        print("\n--- Calculating Audiobox Scores ---")
        audiobox_scores = run_audiobox_evaluation(eval_wav_paths_all, args.batch_size)
        if audiobox_scores:
            df_results['audiobox_ce'] = [s['CE'] for s in audiobox_scores]
            df_results['audiobox_cu'] = [s['CU'] for s in audiobox_scores]
            df_results['audiobox_pc'] = [s['PC'] for s in audiobox_scores]
            df_results['audiobox_pq'] = [s['PQ'] for s in audiobox_scores]
        gc.collect()
        torch.cuda.empty_cache()
        
    if dist_metrics_req:
        print("\n--- Calculating Distributional Scores ---")
        df_baseline = pd.read_csv(args.baseline_csv_path)
        
        kadtk_model_loader = models[args.kadtk_model]
        
        eval_cache_dir = args.eval_wav_dir if args.cache_eval_stats else None

        # For KL, we need paired data. For FAD and KAD, we use the full datasets.
        if 'kl' in args.metrics:
            df_merged = pd.merge(df_eval, df_baseline, on='id', how='inner', suffixes=('_eval', '_baseline'))
            kl_eval_paths = [os.path.join(args.eval_wav_dir, p) for p in df_merged['wav_path_eval'].tolist()]
            kl_baseline_paths = [os.path.join(args.baseline_wav_dir, p) for p in df_merged['wav_path_baseline'].tolist()]
            
            print("Calculating KL...")
            kl_score = run_distributional_evaluation(
                'kl',
                kadtk_model_loader,
                kl_baseline_paths,
                kl_eval_paths,
                args.device,
                args.baseline_wav_dir,
                eval_cache_dir
            )
            df_results['kl'] = kl_score

        # FAD and KAD
        baseline_wav_paths_all = [os.path.join(args.baseline_wav_dir, p) for p in df_baseline['wav_path'].tolist()]
        dist_metrics_to_run = [m for m in args.metrics if m in ['fad', 'kad']]
        
        for metric_name in dist_metrics_to_run:
            print(f"Calculating {metric_name.upper()}...")
            score = run_distributional_evaluation(
                metric_name, 
                kadtk_model_loader, 
                baseline_wav_paths_all, 
                eval_wav_paths_all, 
                args.device,
                args.baseline_wav_dir,
                eval_cache_dir
            )
            df_results[metric_name] = score
        
        del kadtk_model_loader
        gc.collect()
        torch.cuda.empty_cache()

    # --- Save results ---
    output_path = Path(args.output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\nEvaluation complete. Results saved to {output_path}")
    print("\nScore Summary:")
    print(df_results.select_dtypes(include=np.number).mean().to_string())

if __name__ == '__main__':
    main() 