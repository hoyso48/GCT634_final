import os
os.environ['HF_HOME'] = '/workspace/GCT634_final/huggingface'
import torch
import numpy as np
import pandas as pd
import ast
from utils import parse_last_specs, render_from_specs, validate_specs, serialize_specs, clean_name
from scipy.io import wavfile
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from unsloth import FastLanguageModel
from prompt import zeroshot_prompt
from pydx7 import get_algorithm
import argparse
from safetensors import safe_open
from joblib import Parallel, delayed
from tqdm import tqdm

KEY_ORDER = ['algorithm', 'modmatrix', 'outmatrix', 'feedback', 'fixed_freq', 'coarse', 'fine', 'detune', 'transpose', 'ol', 'eg_rate', 'eg_level', 'sensitivity']
PROMPT = zeroshot_prompt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_specs_from_text_prompt(model, tokenizer, text_prompt: str, max_new_tokens: int, temperature, top_p, top_k):
    """
    Generates content using the Hugging Face model (Unsloth).
    """

    messages = [
        {"role": "user", "content": text_prompt}  # Removed unnecessary system prompt
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors = "pt").to(DEVICE)

    streamer = TextStreamer(tokenizer, skip_prompt = True) #Added this for proper output decoding

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            # top_p=top_p, #FIXME
            # top_k=top_k,
            # do_sample=True,  # Enable sampling for temp/top_p/top_k
            streamer=streamer
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def process_caption_row(row_data, model, tokenizer, max_new_tokens, temperature, top_p, top_k, print_response=False):
    """Processes a single row (containing id and caption) to generate patch specs."""
    caption = str(row_data['caption'])
    current_id = row_data['id']
    formatted_llm_prompt = PROMPT.format(prompt=caption)
    patch_data_str = 'FAILED'

    try:
        print(f"Processing ID {current_id} with caption: {caption[:50]}...")
        generated_text = generate_specs_from_text_prompt(model, tokenizer, formatted_llm_prompt, max_new_tokens, temperature, top_p, top_k)

        if print_response:
            print(generated_text)

        try:
            parsed_specs = parse_last_specs(generated_text)
            if validate_specs(parsed_specs, syx_file=f"id_{current_id}", patch_number=-1):
                patch_data_str = serialize_specs(parsed_specs)
                print(f"ID {current_id}: Specs generated and validated.")
            else:
                print(f"ID {current_id}: Specs validation failed.")
                patch_data_str = 'FAILED_VALIDATION'
        except ValueError as e_parse:
            print(f"ID {current_id}: Parsing failed - {e_parse}")
            patch_data_str = 'FAILED_PARSING'
        except Exception as e_val_ser:
            print(f"ID {current_id}: Validation or Serialization error - {e_val_ser}")
            patch_data_str = 'FAILED_PROCESSING'

    except Exception as e_llm:
        print(f"ID {current_id}: Failed to generate specs: {e_llm}")
        patch_data_str = 'FAILED_LLM'

    return current_id, caption, patch_data_str

def generate_and_save_audio_row(row_as_tuple, wav_dir, sample_rate, keys_to_remove, algorithm_predict):
    """Generates and saves a single audio file based on processed spec data."""
    idx, row_data = row_as_tuple
    current_id = row_data['id']
    patch_data_str_input = row_data['generated_patch_data']

    # FIXME: 이전이랑 다르게 더이상 caption 시작하는 게 patch_name가 아닌 경우가 있음
    name = clean_name(row_data['caption'].split(',')[0].strip(), length=10)
    wav_path = f"{current_id}_{name}.wav"

    full_wav_path = os.path.join(wav_dir, wav_path)
    audio_to_save = None
    patch_data = patch_data_str_input

    if isinstance(patch_data_str_input, str) and patch_data_str_input.startswith('FAILED'):
        audio_to_save = np.zeros(sample_rate, dtype=np.int16)
    else:
        try:
            specs = ast.literal_eval(patch_data_str_input)

            # Ensure 'fixed_freq' is present and not in KEYS_TO_REMOVE
            if 'fixed_freq' not in specs:
                specs['fixed_freq'] = [0, 0, 0, 0, 0, 0] #Default fixed freqs

            #Apply KEYS_TO_REMOVE post prediction
            for key in keys_to_remove:
                specs.pop(key, None)  # Remove keys safely
            
            if algorithm_predict:
                specs['algorithm'] = get_algorithm(np.array(specs['modmatrix']), np.array(specs['outmatrix']))

            audio_to_save = render_from_specs(specs, sr=sample_rate)


        except Exception as e_render:
            print(f"ID {current_id}: Failed to render audio - {e_render}. Saving silent audio.")
            audio_to_save = np.zeros(sample_rate, dtype=np.int16)
            patch_data = 'FAILED_RENDER'

    try:
        wavfile.write(full_wav_path, sample_rate, audio_to_save)
    except Exception as e_write:
        print(f"ID {current_id}: Failed to write WAV file {full_wav_path} - {e_write}")
        return current_id, 'FAILED_WRITE', patch_data

    return current_id, wav_path, patch_data

def main():
    parser = argparse.ArgumentParser(description="Generate audio from text captions using a language model and DX7 synthesis.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--caption_csv_path", type=str, required=True, help="Path to the input captions CSV (must contain 'id' and 'caption').")
    parser.add_argument("--output_csv_path", type=str, required=True, help="Path to save the output CSV with generated specs and wav paths.")
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory to save generated WAV files") #made this required
    parser.add_argument("--keys_to_remove", type=str, default="['name', 'has_fixed_freq']", help="Keys to remove from specs")
    parser.add_argument("--precision", type=str, default="fp8", help="Precision for the model (fp8 or bf16)")
    parser.add_argument("--algorithm_predict", action='store_true', help="Predict the algorithm from modmatrix/outmatrix")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument('--max_new_tokens', type=int, default=1024, help="Maximum new tokens for the LLM to generate (default: 1024)")
    parser.add_argument('--sample_rate', type=int, default=48000, help="Sample rate for generated audio.")
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of parallel jobs for audio generation (-1 for all CPUs).")
    parser.add_argument('--print_response', action='store_true', help="Print the response from the LLM.")
    
    # FIXME: 현재 반영안됨!!!
    parser.add_argument("--top_p", type=float, default=0.8, help="Top p for generation")
    parser.add_argument("--top_k", type=int, default=20, help="Top k for generation")

    args = parser.parse_args()
    keys_to_remove = ast.literal_eval(args.keys_to_remove)


    # Create output directory
    os.makedirs(args.wav_dir, exist_ok=True)

     # Load the model and tokenizer
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen3-8B",
        max_seq_length = 768,   # Context length - can be longer, but uses more memory
        load_in_4bit = False,     # 4bit uses much less memory
        load_in_8bit = args.precision == "fp8",    # A bit more accurate, uses 2x memory
        full_finetuning = True, # We have full finetuning now!
    )
    print("Model loaded")
    # Load safetensors weights
    print("Loading weights")
    all_safetensors = [f for f in os.listdir(args.model_path) if f.endswith(".safetensors")]
    state_dict = {}
    for filename in all_safetensors:
        filepath = os.path.join(args.model_path, filename)
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(DEVICE)
    print("Weights loaded")

    # Load data
    df_captions = pd.read_csv(args.caption_csv_path)
    caption_required_columns = ['id', 'caption']
    for col in caption_required_columns:
        if col not in df_captions.columns:
            raise ValueError(f"Required columns {caption_required_columns} not found in {args.caption_csv_path}")
    df_captions = df_captions[caption_required_columns]
    print(f"Found {len(df_captions)} items to process.")

    processed_results_llm = []
    rows_to_process = [row for _, row in df_captions.iterrows()]

    # Process captions to generate specifications
    print("Processing captions with the language model...")
    for row_data in tqdm(rows_to_process, desc="Processing Captions"):
        llm_result = process_caption_row(row_data, model, tokenizer, args.max_new_tokens, args.temperature, args.top_p, args.top_k, args.print_response)
        processed_results_llm.append(llm_result)
    df_after_llm = pd.DataFrame(processed_results_llm, columns=['id', 'caption', 'generated_patch_data'])

    # Generate audio files in parallel
    print("\nGenerating audio files in parallel...")
    audio_processing_args = list(df_after_llm.iterrows())

    audio_gen_results = Parallel(n_jobs=args.n_jobs)(
        delayed(generate_and_save_audio_row)(row_tuple, args.wav_dir, args.sample_rate, ast.literal_eval(args.keys_to_remove), args.algorithm_predict) #Pass keys_to_remove and algorithm_predict
        for row_tuple in tqdm(audio_processing_args, desc="Generating Audio")
    )
    df_audio_results = pd.DataFrame(audio_gen_results, columns=['id', 'wav_path', 'patch_data']) #Correct column names

    # Merge results
    df_merged = pd.merge(
        df_captions,
        df_audio_results,
        on='id',
        how='left'
    )

    # Save the results to CSV
    df_merged.to_csv(args.output_csv_path, index=False) #Added index=False
    print(f"\nProcessing complete. Output saved to {args.output_csv_path}")
    print(f"Generated audio files are in {args.wav_dir}")

if __name__ == "__main__":
    main()

# FIXME
#python generate_specs_finetunedmodel.py --model_path /workspace/GCT634_final/models/Qwen3_8B-fp8-filtered_full-tune --caption_csv_path data/DX7_YAMAHA_test_captions.csv --output_csv_path artifacts/Qwen3_8B-fp8-filtered_full-tune_temp1.0_topP0.8_topK20.csv --wav_dir "artifacts/Qwen3_8B-fp8-filtered_full-tune_temp1.0_topP0.8_topK20" --print_response
# python generate_specs_finetunedmodel.py --model_path /workspace/GCT634_final/models/Qwen3-8B-fp8-filtered_full-tune_no-fixed-freq --caption_csv_path data/DX7_YAMAHA_test_captions.csv --output_csv_path artifacts/Qwen3-8B-fp8-filtered_full-tune_no-fixed-freq_temp1.0_topP0.8_topK20.csv --wav_dir "artifacts/Qwen3-8B-fp8-filtered_full-tune_no-fixed-freq_temp1.0_topP0.8_topK20" --print_response
# python generate_specs_finetunedmodel.py --model_path /workspace/GCT634_final/models/Qwen3-8B-fp8-filtered_resp-only --caption_csv_path data/DX7_YAMAHA_test_captions.csv --output_csv_path artifacts/Qwen3-8B-fp8-filtered_resp-only_temp1.0_topP0.8_topK20.csv --wav_dir "artifacts/Qwen3-8B-fp8-filtered_resp-only_temp1.0_topP0.8_topK20" --print_response
# python generate_specs_finetunedmodel.py --model_path /workspace/GCT634_final/models/Qwen3-8B-fp8-filtered_resp-only_no-fixed-freq --caption_csv_path data/DX7_YAMAHA_test_captions.csv --output_csv_path artifacts/Qwen3-8B-fp8-filtered_resp-only_no-fixed-freq_temp1.0_topP0.8_topK20.csv --wav_dir "artifacts/Qwen3-8B-fp8-filtered_resp-only_no-fixed-freq_temp1.0_topP0.8_topK20" --print_response

