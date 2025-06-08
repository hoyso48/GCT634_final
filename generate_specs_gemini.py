from google import genai
import pandas as pd
from tqdm import tqdm
import argparse
import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os
import numpy as np
from scipy.io.wavfile import write
import ast
from joblib import Parallel, delayed

from gemini_api_key import API_KEY  # Make sure this file exists and API_KEY is defined
from prompt import zeroshot_prompt # Assuming zeroshot_prompt is directly importable
from utils import parse_last_specs, validate_specs, serialize_specs, render_from_specs, clean_name

# Initialize Gemini Client
CLIENT = genai.Client(api_key=API_KEY)
PROMPT = zeroshot_prompt

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(5))
async def generate_specs_from_text_prompt(client_instance, model_name: str, formatted_prompt: str):
    """
    Generates content using the Gemini API based on a text prompt.
    """
    response = await client_instance.aio.models.generate_content(
        model=model_name,
        contents=[formatted_prompt] 
    )
    return response.text

async def process_caption_row(client_instance, row_data, model_name: str, print_response=False):
    """
    Processes a single row (containing id and caption) to generate patch specs.
    """
    caption = str(row_data['caption']) # Ensure caption is a string
    current_id = row_data['id']
    
    formatted_prompt = PROMPT.format(prompt=caption)
    patch_data_str = 'FAILED'
    
    try:
        print(f"Processing ID {current_id} with caption: {caption[:50]}...")
        generated_text = await generate_specs_from_text_prompt(client_instance, model_name, formatted_prompt)
        if print_response:
            print(generated_text)
        try:
            parsed_specs = parse_last_specs(generated_text)
            # For validate_specs, syx_file and patch_number are not directly relevant here
            # Pass placeholder values or adapt validate_specs if needed.
            if validate_specs(parsed_specs, syx_file=f"id_{current_id}", patch_number=-1):
                patch_data_str = serialize_specs(parsed_specs)
                print(f"ID {current_id}: Specs generated and validated.")
            else:
                print(f"ID {current_id}: Specs validation failed.")
                patch_data_str = 'FAILED_VALIDATION' # More specific failure reason
        except ValueError as e_parse:
            print(f"ID {current_id}: Parsing failed - {e_parse}")
            patch_data_str = 'FAILED_PARSING'
        except Exception as e_val_ser: # Catch other potential errors during validation/serialization
             print(f"ID {current_id}: Validation or Serialization error - {e_val_ser}")
             patch_data_str = 'FAILED_PROCESSING'

    except Exception as e_gemini:
        error_message = f"ID {current_id}: Failed to generate specs after multiple retries: {e_gemini}"
        # Check if the exception is a RetryError and try to get more details from the cause
        if hasattr(e_gemini, '__cause__') and e_gemini.__cause__:
            error_message += f"\\nRoot cause: {e_gemini.__cause__}"
        print(error_message)
        patch_data_str = 'FAILED_API'
        
    return current_id, caption, patch_data_str

# New function for parallel audio generation
def generate_and_save_audio_row(row_as_tuple, wav_dir, sample_rate):
    """
    Generates and saves a single audio file based on processed spec data from a row tuple (idx, Series).
    Returns id, wav_filename_status, and potentially updated patch_data_status.
    """
    idx, row_data = row_as_tuple # row_data is a Pandas Series here
    
    current_id = row_data['id']
    patch_data_str_input = row_data['generated_patch_data'] 
    
    wav_name = clean_name(row_data['caption'].split(',')[0].strip(), length=10)
    wav_path = f"{current_id}_{wav_name}.wav"
    full_wav_path = os.path.join(wav_dir, wav_path)
    
    audio_to_save = None
    # Initialize final_patch_data_status with the input, it might change if rendering fails
    patch_data = patch_data_str_input 

    if patch_data_str_input.startswith('FAILED'):
        audio_to_save = np.zeros(sample_rate, dtype=np.int16)
    else:
        try:
            specs_dict = ast.literal_eval(patch_data_str_input) 
            audio_to_save = render_from_specs(specs_dict, sr=sample_rate)
        except Exception:
            audio_to_save = np.zeros(sample_rate, dtype=np.int16)
            patch_data = 'FAILED_RENDER' 

    write(full_wav_path, sample_rate, audio_to_save)

    return current_id, wav_path, patch_data

async def main():
    parser = argparse.ArgumentParser(description="Generate DX7 specs from text captions using Gemini and then render audio.")
    parser.add_argument('--model', type=str, default='gemini-2.5-flash-preview-05-20', help="Gemini model name to use.")
    parser.add_argument('--caption_csv_path', type=str, required=True, help="Path to the input captions CSV (must contain 'id' and 'caption').")
    parser.add_argument('--output_csv_path', type=str, required=True, help="Path to save the output CSV with generated specs and wav paths.")
    parser.add_argument('--wav_dir', type=str, default='data/generated/gemini', help="Directory to save the generated WAV files.")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for processing captions with Gemini.")
    parser.add_argument('--sample_rate', type=int, default=48000, help="Sample rate for generated audio.")
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of parallel jobs for audio generation (-1 for all CPUs).")
    parser.add_argument('--print_response', action='store_true', help="Print the response from the LLM.")
    args = parser.parse_args()

    # Ensure wav_dir exists
    if not os.path.exists(args.wav_dir):
        os.makedirs(args.wav_dir, exist_ok=True)

    # Load data
    df_captions = pd.read_csv(args.caption_csv_path)

    caption_required_columns = ['id', 'caption']
    for col in caption_required_columns:
        if col not in df_captions.columns:
            raise ValueError(f"Required columns {caption_required_columns} not found in {args.caption_csv_path}")

    df_captions = df_captions[caption_required_columns]

    print(f"Found {len(df_captions)} items to process.")

    processed_results_api = [] # To store (id, caption, patch_data_str)
    
    rows_to_process = [row for _, row in df_captions.iterrows()]

    for i in tqdm(range(0, len(rows_to_process), args.batch_size), desc="Processing caption batches for API"):
        batch_rows_data = rows_to_process[i:i + args.batch_size]
        tasks = [process_caption_row(CLIENT, row_data, args.model, args.print_response) for row_data in batch_rows_data]
        
        batch_api_results = await asyncio.gather(*tasks)
        processed_results_api.extend(batch_api_results)
        print(f"Processed API batch {i // args.batch_size + 1}/{(len(rows_to_process) + args.batch_size - 1) // args.batch_size}")

    # Create a DataFrame from results
    df_after_api = pd.DataFrame(processed_results_api, columns=['id', 'caption', 'generated_patch_data'])

    print(f"\nGenerating and saving {len(df_after_api)} audio files in parallel...")
    
    audio_processing_args = list(df_after_api.iterrows()) # list of (index, Series) tuples

    audio_gen_results = Parallel(n_jobs=args.n_jobs)(
        delayed(generate_and_save_audio_row)(row_tuple, args.wav_dir, args.sample_rate) 
        for row_tuple in tqdm(audio_processing_args, desc="Generating Audio")
    )

    df_audio_results = pd.DataFrame(audio_gen_results, columns=['id', 'wav_path', 'patch_data'])

    df_merged = pd.merge(
        df_captions, 
        df_audio_results, # Contains 'id', 'wav_path', 'patch_data'
        on='id', 
        how='left'
    )

    df_merged.to_csv(args.output_csv_path)
    print(f"\nProcessing complete. Output saved to {args.output_csv_path}")
    print(f"Generated audio files are in {args.wav_dir}")

if __name__ == '__main__':
    asyncio.run(main())
