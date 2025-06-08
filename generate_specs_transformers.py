import pandas as pd
from tqdm import tqdm
import argparse
import os
import numpy as np
from scipy.io.wavfile import write
import ast
from joblib import Parallel, delayed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import zeroshot_prompt 
from utils import parse_last_specs, validate_specs, serialize_specs, render_from_specs, clean_name

PROMPT = zeroshot_prompt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None
TOKENIZER = None

def generate_specs_from_text_prompt(model, tokenizer, text_prompt: str, max_new_tokens: int):
    """
    Generates content using the Hugging Face model.
    """
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}, #system prompt, can be customized
        {"role": "user", "content": text_prompt}
    ]
    
    try:
        prompt_for_model = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([prompt_for_model], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                # Common generation parameters (can be customized or made arguments)
                # num_beams=1, # Use greedy search by default for speed
                # do_sample=False, 
                # temperature=0.7, 
                # top_p=0.9
            )
        
        # Decode only the newly generated tokens
        input_ids_length = model_inputs.input_ids.shape[1]
        generated_token_ids = generated_ids[0, input_ids_length:]
        response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        return response_text
    except Exception as e:
        print(f"Error during model generation: {e}")
        raise # Re-raise to trigger tenacity retry

def process_caption_row(row_data, model, tokenizer, max_new_tokens: int, print_response=False):
    """
    Processes a single row (containing id and caption) to generate patch specs using Hugging Face model.
    """
    caption = str(row_data['caption'])
    current_id = row_data['id']
    
    # The zeroshot_prompt already contains the detailed structure.
    # We pass the user's caption into the {prompt} placeholder of zeroshot_prompt.
    formatted_llm_prompt = PROMPT.format(prompt=caption)
    patch_data_str = 'FAILED' 
    
    try:
        print(f"Processing ID {current_id} with caption: {caption}")
        generated_text = generate_specs_from_text_prompt(model, tokenizer, formatted_llm_prompt, max_new_tokens)
        if print_response:
            print(generated_text)
        if not generated_text or generated_text.strip() == "":
            print(f"ID {current_id}: Empty response from LLM.")
            patch_data_str = 'FAILED_EMPTY_RESPONSE'
        else:
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

    except Exception as e_llm: # Covers LLM call failures after retries
        print(f"ID {current_id}: Failed to generate specs after multiple retries: {e_llm}")
        patch_data_str = 'FAILED_LLM' # Renamed from FAILED_API
        
    return current_id, caption, patch_data_str

def generate_and_save_audio_row(row_as_tuple, wav_dir, sample_rate):
    idx, row_data = row_as_tuple
    current_id = row_data['id']
    patch_data_str_input = row_data['generated_patch_data']
    
    wav_name = clean_name(row_data['caption'].split(',')[0].strip(), length=10)
    wav_path = f"{current_id}_{wav_name}.wav"
    full_wav_path = os.path.join(wav_dir, wav_path)
    audio_to_save = None
    patch_data = patch_data_str_input

    if isinstance(patch_data_str_input, str) and patch_data_str_input.startswith('FAILED'):
        audio_to_save = np.zeros(sample_rate, dtype=np.int16)
    else:
        try:
            # Ensure patch_data_str_input is a string before ast.literal_eval
            if not isinstance(patch_data_str_input, str):
                 raise ValueError(f"Patch data for ID {current_id} is not a string: {patch_data_str_input}")
            specs_dict = ast.literal_eval(patch_data_str_input)
            audio_to_save = render_from_specs(specs_dict, sr=sample_rate)
        except Exception as e_render:
            print(f"ID {current_id}: Failed to render audio - {e_render}. Saving silent audio.")
            audio_to_save = np.zeros(sample_rate, dtype=np.int16)
            patch_data = 'FAILED_RENDER'
            if isinstance(patch_data_str_input, str) and not patch_data_str_input.startswith('FAILED'): # Log original if it wasn't already failed
                 print(f"Original patch data for {current_id} that failed render: {patch_data_str_input[:200]}...")

    try:
        write(full_wav_path, sample_rate, audio_to_save)
    except Exception as e_write:
        print(f"ID {current_id}: Failed to write WAV file {full_wav_path} - {e_write}")
        return current_id, 'FAILED_WRITE', patch_data # wav_path is failure status

    return current_id, wav_path, patch_data


def main():
    parser = argparse.ArgumentParser(description="Generate DX7 specs from text captions using a Hugging Face  model and then render audio.")
    parser.add_argument('--model_name_or_path', type=str, default="Qwen/Qwen2.5-14B-Instruct", help="Hugging Face model name or path (default: Qwen/Qwen2.5-7B-Instruct).")
    parser.add_argument('--max_new_tokens', type=int, default=1024, help="Maximum new tokens for the LLM to generate (default: 1024)")
    parser.add_argument('--caption_csv_path', type=str, required=True, help="Path to the input captions CSV (must contain 'id' and 'caption').")
    parser.add_argument('--output_csv_path', type=str, required=True, help="Path to save the output CSV with generated specs and wav paths.")
    parser.add_argument('--wav_dir', type=str, default='data/generated_wav/qwen', help="Directory to save the generated WAV files.")
    parser.add_argument('--sample_rate', type=int, default=48000, help="Sample rate for generated audio.")
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of parallel jobs for audio generation (-1 for all CPUs).")
    parser.add_argument('--print_response', action='store_true', help="Print the response from the LLM.")

    args = parser.parse_args()

    print(f"Loading model: {args.model_name_or_path} on device: {DEVICE} (using torch_dtype=torch.bfloat16)")
    try:
        TOKENIZER = AutoTokenizer.from_pretrained(args.model_name_or_path)
        MODEL = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        if TOKENIZER.pad_token is None:
            TOKENIZER.pad_token = TOKENIZER.eos_token
            MODEL.config.pad_token_id = MODEL.config.eos_token_id
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    if not os.path.exists(args.wav_dir):
        os.makedirs(args.wav_dir, exist_ok=True)

    df_captions = pd.read_csv(args.caption_csv_path)
    caption_required_columns = ['id', 'caption']
    for col in caption_required_columns:
        if col not in df_captions.columns:
            raise ValueError(f"Required columns {caption_required_columns} not found in {args.caption_csv_path}")
    df_captions = df_captions[caption_required_columns]

    print(f"Found {len(df_captions)} items to process.")

    processed_results_llm = []
    rows_to_process = [row for _, row in df_captions.iterrows()]

    for row_data in tqdm(rows_to_process, desc="Processing captions for LLM"):
        llm_result = process_caption_row(row_data, MODEL, TOKENIZER, args.max_new_tokens, args.print_response)
        processed_results_llm.append(llm_result)

    df_after_llm = pd.DataFrame(processed_results_llm, columns=['id', 'caption', 'generated_patch_data'])

    print(f"\nGenerating and saving {len(df_after_llm)} audio files in parallel...")
    audio_processing_args = list(df_after_llm.iterrows())

    audio_gen_results = Parallel(n_jobs=args.n_jobs)(
        delayed(generate_and_save_audio_row)(row_tuple, args.wav_dir, args.sample_rate)
        for row_tuple in tqdm(audio_processing_args, desc="Generating Audio")
    )

    df_audio_results = pd.DataFrame(audio_gen_results, columns=['id', 'wav_path', 'patch_data'])
    
    df_merged = pd.merge(
        df_captions,
        df_audio_results,
        on='id',
        how='left'
    )

    df_merged.to_csv(args.output_csv_path)
    print(f"\nProcessing complete. Output saved to {args.output_csv_path}")
    print(f"Generated audio files are in {args.wav_dir}")

if __name__ == '__main__':
    main()