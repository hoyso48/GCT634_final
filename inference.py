import os
os.environ['HF_HOME'] = '/workspace/GCT634_final/huggingface'
import torch
import numpy as np
import pandas as pd
import ast
from utils import parse_last_specs, render_from_specs, validate_specs
from scipy.io import wavfile
from transformers import TextStreamer
from unsloth import FastLanguageModel
from prompt import zeroshot_prompt
from pydx7 import get_algorithm
import argparse
from safetensors import safe_open

KEY_ORDER = ['algorithm', 'modmatrix', 'outmatrix', 'feedback', 'fixed_freq', 'coarse', 'fine', 'detune', 'transpose', 'ol', 'eg_rate', 'eg_level', 'sensitivity']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated WAV files")
    parser.add_argument("--keys_to_remove", type=str, default="['name', 'has_fixed_freq']", help="Keys to remove from specs")
    parser.add_argument("--precision", type=str, default="fp8")
    parser.add_argument("--algorithm_predict", action='store_true')
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top p for generation")
    parser.add_argument("--top_k", type=int, default=20, help="Top k for generation")
    
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    OUTPUT_DIR = args.output_dir
    KEYS_TO_REMOVE = ast.literal_eval(args.keys_to_remove)
    PRECISION = args.precision
    ALGORITHM_PREDICT = args.algorithm_predict
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    TOP_K = args.top_k

    # Create output directory based on model name, temp, top_p, and top_k
    model_name = os.path.basename(MODEL_PATH)  # Extract "Qwen3-8B-fp8-filtered_full-tune"
    output_sub_dir = f"{model_name}_temp{TEMPERATURE}_topP{TOP_P}_topK{TOP_K}"
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, output_sub_dir)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen3-8B",
        max_seq_length = 768,   # Context length - can be longer, but uses more memory
        load_in_4bit = False,     # 4bit uses much less memory
        load_in_8bit = PRECISION == "fp8",    # A bit more accurate, uses 2x memory
        full_finetuning = True, # We have full finetuning now!
        # token = "hf_...",      # use one if using gated models
    )

    # Load safetensors weights
    all_safetensors = [f for f in os.listdir(MODEL_PATH) if f.endswith(".safetensors")]
    state_dict = {}
    for filename in all_safetensors:
        filepath = os.path.join(MODEL_PATH, filename)
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys(): 
                state_dict[key] = f.get_tensor(key)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to("cuda")

    # Load the test dataset
    test_caption = pd.read_csv("data/DX7_YAMAHA_test_captions.csv", index_col=0)
    test_data = pd.read_csv("data/DX7_YAMAHA_test.csv", index_col=0)
    test_df = pd.merge(test_data, test_caption[['id', 'caption']], on='id', how='left')
    test_filter = (test_df['inaudible'] == False) & ~test_df['name'].str.contains('NULL')
    test_filter &= ~test_df['has_fixed_freqs']

    test_df = test_df[test_filter]

    # Inference loop
    for index, row in test_df.iterrows():
        caption = row['caption']
        print(f"Processing caption: {caption}")

        # Construct the prompt
        messages = [
            {"role" : "user", "content" : zeroshot_prompt.format(prompt=caption)}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True, # Must add for generation
            enable_thinking = False, # Disable thinking
        )

        # Generate the specs
        with torch.no_grad():
            output = model.generate(
                **tokenizer(text, return_tensors = "pt").to("cuda"),
                max_new_tokens = 1024, # Increase for longer outputs!
                temperature = TEMPERATURE, #, top_p = 0.8, top_k = 20, # For non thinking
                # top_p = TOP_P,
                # top_k = TOP_K,
                # do_sample = True,
                streamer = TextStreamer(tokenizer, skip_prompt = True),
            )

        # Extract and validate the specs
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        specs = parse_last_specs(decoded_output)

        # Ensure 'fixed_freq' is present
        if 'fixed_freq' not in specs:
            specs['fixed_freq'] = [0, 0, 0, 0, 0, 0]

        try:
            validate_specs(specs)
        except Exception as e:
            print(f"Validation failed for caption '{caption}': {e}")
            continue

        # Render and save the audio
        if 'fixed_freq' in KEYS_TO_REMOVE:
            specs['fixed_freq'] = [0,0,0,0,0,0]
        
        if ALGORITHM_PREDICT:
            specs['algorithm'] = get_algorithm(np.array(specs['modmatrix']), np.array(specs['outmatrix']))

        audio = render_from_specs(specs)

        # Create filename with index and name
        name = row['name'].replace(" ", "_").replace("/", "_")  # Sanitize name
        output_filename = os.path.join(OUTPUT_DIR, f"audio_{index}_{name}.wav")
        wavfile.write(output_filename, 44100, audio)
        print(f"Saved audio to: {output_filename}")

    print("Inference complete.")

# python inference.py --model_path /workspace/GCT634_final/models/Qwen3_8B-fp8-filtered_full-tune --output_dir "artifacts" --keys_to_remove "['name', 'has_fixed_freq']" --precision "fp8" --temperature 1.0 --top_p 0.8 --top_k 20 
# python inference.py --model_path /workspace/GCT634_final/models/Qwen3-8B-fp8-filtered_full-tune_no-fixed-freq --output_dir "artifacts" --keys_to_remove "['name', 'has_fixed_freq']" --precision "fp8" --temperature 1.0 --top_p 0.8 --top_k 20 


# python inference.py --model_path /workspace/GCT634_final/models/Qwen3-8B-fp8-filtered_resp-only --output_dir "artifacts" --keys_to_remove "['name', 'has_fixed_freq']" --precision "fp8" --temperature 1.0 --top_p 0.8 --top_k 20 

# python inference.py --model_path /workspace/GCT634_final/models/Qwen3-8B-fp8-filtered_resp-only_no-fixed-freq --output_dir "artifacts" --keys_to_remove "['name', 'has_fixed_freq']" --precision "fp8" --temperature 1.0 --top_p 0.8 --top_k 20 

