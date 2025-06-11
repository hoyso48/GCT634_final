from clap_score import compute_clap_score, load_clap_model
from audiobox_aesthetics.infer import initialize_predictor

import argparse
import pandas as pd
from tqdm import tqdm
import os
import torch
import gc

def evaluate(clap_model, audiobox_model, df, df_captions, wav_dir, batch_size=32):

    df = pd.merge(df, df_captions, on='id', how='inner')
    print(df.head())
    print(df.columns)
    
    wav_paths = [os.path.join(wav_dir, wav_path) for wav_path in df['wav_path'].tolist()]
    clap_scores = compute_clap_score(clap_model, df.caption.tolist(), wav_paths, batch_size=batch_size)
    clap_scores_synthesized = compute_clap_score(clap_model, ['Synthesized ' + caption for caption in df.caption.tolist()], wav_paths, batch_size=batch_size)

    audiobox_scores = []
    for i in range(0, len(wav_paths), batch_size):
        audiobox_scores.extend(audiobox_model.forward([{"path":p} for p in wav_paths[i:i+batch_size]]))
    if i+batch_size < len(wav_paths):
        audiobox_scores.extend(audiobox_model.forward([{"path":p} for p in wav_paths[i+batch_size:]]))

    audiobox_ces = [score['CE'] for score in audiobox_scores]
    audiobox_cus = [score['CU'] for score in audiobox_scores]
    audiobox_pcs = [score['PC'] for score in audiobox_scores]
    audiobox_pqs = [score['PQ'] for score in audiobox_scores]

    df_scores = pd.DataFrame({
        'id': df.id.tolist(),
        'wav_path': df.wav_path.tolist(),
        'caption': df.caption.tolist(),
        'clap_score': clap_scores,
        'clap_score_synthesized': clap_scores_synthesized,
        'audiobox_ce': audiobox_ces,
        'audiobox_cu': audiobox_cus,
        'audiobox_pc': audiobox_pcs,
        'audiobox_pq': audiobox_pqs
    })

    return df_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path", type=str, required=True)
    parser.add_argument("--caption_csv_path", type=str, required=True)
    parser.add_argument("--output_csv_path", type=str, required=True)
    parser.add_argument("--wav_dir", type=str, default='data/wav')
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    df = pd.read_csv(args.data_csv_path, index_col=0)

    df_captions = pd.read_csv(args.caption_csv_path, index_col=0)

    data_required_columns = ['id', 'wav_path']
    caption_required_columns = ['id', 'caption']
    for col in data_required_columns:
        if col not in df.columns:
            raise ValueError(f"Required columns {data_required_columns} not found in {args.data_csv_path}")
    for col in caption_required_columns:
        if col not in df_captions.columns:
            raise ValueError(f"Required columns {caption_required_columns} not found in {args.caption_csv_path}")

    df = df[data_required_columns]
    df_captions = df_captions[caption_required_columns]

    clap_model = load_clap_model()
    audiobox_model = initialize_predictor()

    result = evaluate(clap_model, audiobox_model, df, df_captions, args.wav_dir, args.batch_size)
    print(result[['clap_score', 'clap_score_synthesized', 'audiobox_ce', 'audiobox_cu', 'audiobox_pc', 'audiobox_pq']].describe())
    result.to_csv(args.output_csv_path)


# python evaluate.py --data_csv_path artifacts/Qwen3_8B-fp8-filtered_full-tune_temp1.0_topP0.8_topK20.csv --caption_csv_path data/DX7_YAMAHA_captions_test.csv --output_csv_path artifacts/Qwen3_8B-fp8-filtered_full-tune_temp1.0_topP0.8_topK20_score.csv --wav_dir artifacts/Qwen3_8B-fp8-filtered_full-tune_temp1.0_topP0.8_topK20


#python generate_specs_finetunedmodel.py --model_path /workspace/GCT634_final/models/Qwen3_8B-fp8-filtered_full-tune --caption_csv_path data/DX7_YAMAHA_test_captions.csv --output_csv_path artifacts/Qwen3_8B-fp8-filtered_full-tune_temp1.0_topP0.8_topK20.csv --wav_dir "artifacts/Qwen3_8B-fp8-filtered_full-tune_temp1.0_topP0.8_topK20" --print_response
