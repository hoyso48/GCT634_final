from utils import render_from_specs
from scipy.io.wavfile import write
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
import argparse


def process_row(row_tuple, sr=48000, n=60, v=100, out_scale=1.0, wav_dir='data/wav'):
    index, row_data = row_tuple  # Unpack the tuple from iterrows()
    wav_path = wav_dir + '/' + row_data['wav_path']
    wav_dir = os.path.dirname(wav_path)
    
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir, exist_ok=True)
        
    patch_data_processed = eval(row_data['patch_data'])

    audio = render_from_specs(patch_data_processed, sr=sr, n=n, v=v, out_scale=out_scale)
    write(wav_path, sr, audio)
    return f"Processed {wav_path}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='data/DX7_YAMAHA_deduplicated.csv')
    parser.add_argument('--wav_dir', type=str, default='data/wav')
    parser.add_argument('--sr', type=int, default=48000)
    parser.add_argument('--n', type=int, default=60)
    parser.add_argument('--v', type=int, default=100)
    parser.add_argument('--out_scale', type=float, default=1.0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    print(f"Starting parallel processing for {len(df)} items...")

    results = Parallel(n_jobs=-1)(
        delayed(process_row)(row_tuple, sr=args.sr, n=args.n, v=args.v, out_scale=args.out_scale, wav_dir=args.wav_dir) for row_tuple in tqdm(df.iterrows(), total=len(df), desc="Generating audio")
    )

    print(f"\nFinished processing {len(results)} items.")