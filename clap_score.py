import os
import requests
from tqdm import tqdm
import torch
import numpy as np
import laion_clap
from clap_module.factory import load_state_dict
import librosa
import pyloudnorm as pyln

# following documentation from https://github.com/LAION-AI/CLAP
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def clap_score(id2text, audio_path, audio_files_extension='.wav', clap_model='630k-audioset-fusion-best.pt'):
    """
    Cosine similarity is computed between the LAION-CLAP text embedding of the given prompt and 
    the LAION-CLAP audio embedding of the generated audio. LION-CLAP: https://github.com/LAION-AI/CLAP
    
    This evaluation script assumes that audio_path files are identified with the ids in id2text.
    
    clap_score() evaluates all ids in id2text.

    GPU-based computation.

    Select one of the following models from https://github.com/LAION-AI/CLAP:
        - music_speech_audioset_epoch_15_esc_89.98.pt (used by musicgen)
        - music_audioset_epoch_15_esc_90.14.pt
        - music_speech_epoch_15_esc_89.25.pt
        - 630k-audioset-fusion-best.pt (our default, with "fusion" to handle longer inputs)

    Params:
    -- id2text: dictionary with the mapping between id (generated audio filenames in audio_path) 
                and text (prompt used to generate audio). clap_score() evaluates all ids in id2text.
    -- audio_path: path where the generated audio files to evaluate are available.
    -- audio_files_extension: files extension (default .wav) in eval_path.
    -- clap_model: choose one of the above clap_models (default: '630k-audioset-fusion-best.pt').
    Returns:
    -- CLAP-LION score
    """
    # load model
    if clap_model == 'music_speech_audioset_epoch_15_esc_89.98.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt'
        clap_path = 'load/clap_score/music_speech_audioset_epoch_15_esc_89.98.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
    elif clap_model == 'music_audioset_epoch_15_esc_90.14.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'
        clap_path = 'load/clap_score/music_audioset_epoch_15_esc_90.14.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
    elif clap_model == 'music_speech_epoch_15_esc_89.25.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt'
        clap_path = 'load/clap_score/music_speech_epoch_15_esc_89.25.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device='cuda')
    elif clap_model == '630k-audioset-fusion-best.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt'
        clap_path = 'load/clap_score/630k-audioset-fusion-best.pt'
        model = laion_clap.CLAP_Module(enable_fusion=True, device='cuda')
    else:
        raise ValueError('clap_model not implemented')

    # download clap_model if not already downloaded
    if not os.path.exists(clap_path):
        print('Downloading ', clap_model, '...')
        os.makedirs(os.path.dirname(clap_path), exist_ok=True)

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(clap_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
                for data in response.iter_content(chunk_size=8192):
                    file.write(data)
                    progress_bar.update(len(data))

    # fixing CLAP-LION issue, see: https://github.com/LAION-AI/CLAP/issues/118
    pkg = load_state_dict(clap_path)
    pkg.pop('text_branch.embeddings.position_ids', None)
    model.model.load_state_dict(pkg)
    model.eval()

    if not os.path.isdir(audio_path):        
        raise ValueError('audio_path does not exist')

    if id2text:   
        print('[EXTRACTING TEXT EMBEDDINGS] ')
        batch_size = 64
        text_emb = {}
        for i in tqdm(range(0, len(id2text), batch_size)):
            batch_ids = list(id2text.keys())[i:i+batch_size]
            batch_texts = [id2text[id] for id in batch_ids]
            with torch.no_grad():
                embeddings = model.get_text_embedding(batch_texts, use_tensor=True)
            for id, emb in zip(batch_ids, embeddings):
                text_emb[id] = emb

    else:
        raise ValueError('Must specify id2text')

    print('[EVALUATING GENERATIONS] ', audio_path)
    score = 0
    count = 0
    for id in tqdm(id2text.keys()):
        file_path = os.path.join(audio_path, str(id)+audio_files_extension)
        with torch.no_grad():
            audio, _ = librosa.load(file_path, sr=48000, mono=True) # sample rate should be 48000
            audio = pyln.normalize.peak(audio, -1.0)
            audio = audio.reshape(1, -1) # unsqueeze (1,T)
            audio = torch.from_numpy(int16_to_float32(float32_to_int16(audio))).float()
            audio_embeddings = model.get_audio_embedding_from_data(x = audio, use_tensor=True)
        cosine_sim = torch.nn.functional.cosine_similarity(audio_embeddings, text_emb[id].unsqueeze(0), dim=1, eps=1e-8)[0]
        score += cosine_sim
        count += 1

    return score / count if count > 0 else 0


# New functions added below

def load_clap_model(clap_model_name='630k-audioset-fusion-best.pt', device='cuda'):
    """
    Loads a LAION-CLAP model.

    Args:
        clap_model_name (str): Name of the CLAP model to load.
            Options: 'music_speech_audioset_epoch_15_esc_89.98.pt', 
                     'music_audioset_epoch_15_esc_90.14.pt',
                     'music_speech_epoch_15_esc_89.25.pt',
                     '630k-audioset-fusion-best.pt'
        device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
        laion_clap.CLAP_Module: The loaded CLAP model.
    """
    if clap_model_name == 'music_speech_audioset_epoch_15_esc_89.98.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt'
        clap_path = 'load/clap_score/music_speech_audioset_epoch_15_esc_89.98.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device=device)
    elif clap_model_name == 'music_audioset_epoch_15_esc_90.14.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'
        clap_path = 'load/clap_score/music_audioset_epoch_15_esc_90.14.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device=device)
    elif clap_model_name == 'music_speech_epoch_15_esc_89.25.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt'
        clap_path = 'load/clap_score/music_speech_epoch_15_esc_89.25.pt'
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device=device)
    elif clap_model_name == '630k-audioset-fusion-best.pt':
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt'
        clap_path = 'load/clap_score/630k-audioset-fusion-best.pt'
        model = laion_clap.CLAP_Module(enable_fusion=True, device=device) 
    else:
        raise ValueError(f"clap_model_name '{clap_model_name}' not implemented")

    if not os.path.exists(clap_path):
        print('Downloading ', clap_model_name, '...')
        os.makedirs(os.path.dirname(clap_path), exist_ok=True)
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(clap_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {clap_model_name}") as progress_bar:
                for data in response.iter_content(chunk_size=8192):
                    file.write(data)
                    progress_bar.update(len(data))

    # Try to load with device argument if available, otherwise default
    try:
        pkg = load_state_dict(clap_path, device=device)
    except TypeError: # Older version of factory.py might not support device arg
        pkg = load_state_dict(clap_path)
        
    if 'text_branch.embeddings.position_ids' in pkg:
        pkg.pop('text_branch.embeddings.position_ids', None)
    model.model.load_state_dict(pkg)
    model.eval()
    model.to(device) 
    return model

def compute_clap_score(model, text_list, audio_path_list, batch_size=32, device='cuda'):
    """
    Computes CLAP scores for a list of text prompts and corresponding audio files using a pre-loaded model.

    Args:
        model (laion_clap.CLAP_Module): The pre-loaded CLAP model.
        text_list (list of str): A list of text prompts.
        audio_path_list (list of str): A list of paths to audio files.
        batch_size (int): Batch size for inference.
        audio_sample_rate (int): The sample rate to which audio will be resampled.
        device (str): Device to perform computation on ('cuda' or 'cpu').

    Returns:
        list of float: A list of CLAP scores. Nan for scores that could not be computed.
    """
    if not text_list or not audio_path_list:
        # Return empty list or raise error if appropriate for evaluate.py
        return [] 
    if len(text_list) != len(audio_path_list):
        # Return empty list or raise error
        print("Warning: text_list and audio_path_list lengths differ. Returning empty score list.")
        return [] 

    model.eval() 
    model.to(device)

    all_scores = []
    
    text_embeddings_list = []
    # print('[EXTRACTING TEXT EMBEDDINGS FOR COMPUTE_CLAP_SCORE]') # Optional: for debugging
    for i in tqdm(range(0, len(text_list), batch_size), desc="Text Embedding Batches (compute_clap_score)", leave=False):
        batch_texts = text_list[i:i+batch_size]
        with torch.no_grad():
            embeddings = model.get_text_embedding(batch_texts, use_tensor=True)
            text_embeddings_list.append(embeddings.to(device)) 
    
    if not text_embeddings_list: # Should not happen if text_list is not empty
        return [float('nan')] * len(text_list)

    text_embeddings_tensor = torch.cat(text_embeddings_list, dim=0)

    # print('[EVALUATING GENERATIONS FOR COMPUTE_CLAP_SCORE]') # Optional: for debugging
    for i in tqdm(range(0, len(audio_path_list), batch_size), desc="Audio Batch Evaluation (compute_clap_score)", leave=False):
        batch_audio_paths = audio_path_list[i:i+batch_size]
        batch_text_embeddings = text_embeddings_tensor[i:i+batch_size]
        
        audio_batch_tensors = []
        current_batch_actual_paths = [] # Keep track of paths for which audio was loaded
        
        max_len = 0
        
        for audio_idx, audio_path in enumerate(batch_audio_paths):
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found {audio_path}. Will assign NaN score for this item.")
                # We'll add a placeholder for score later, need to keep track of original batch size
                continue # Skip loading for this path

            try:
                audio, _ = librosa.load(audio_path, sr=48000, mono=True)
                audio = pyln.normalize.peak(audio, -1.0) 
                audio_processed = int16_to_float32(float32_to_int16(audio)) 
                audio_tensor = torch.from_numpy(audio_processed).float().unsqueeze(0).to(device) 
                audio_batch_tensors.append(audio_tensor)
                current_batch_actual_paths.append(audio_path) # Corresponding path
                if audio_tensor.shape[1] > max_len:
                    max_len = audio_tensor.shape[1]
            except Exception as e:
                print(f"Error loading or processing audio {audio_path}: {e}. Will assign NaN score.")
                # Also skip this if loading fails

        # Pad audio tensors in the batch
        padded_audio_batch = []
        if audio_batch_tensors: # Only pad if we successfully loaded some audios
            for audio_tensor in audio_batch_tensors:
                padding_needed = max_len - audio_tensor.shape[1]
                if padding_needed > 0:
                    padded_tensor = torch.nn.functional.pad(audio_tensor, (0, padding_needed))
                else:
                    padded_tensor = audio_tensor
                padded_audio_batch.append(padded_tensor)
        
        batch_scores = [float('nan')] * len(batch_audio_paths) # Initialize scores for the current batch with NaN

        if padded_audio_batch: # If there are any audios to process in this batch
            audio_batch_stacked = torch.cat(padded_audio_batch, dim=0).to(device)

            # Align text embeddings for successfully loaded audios
            # This is tricky. We need to match text_embeddings to successfully loaded audios.
            # The text_embeddings are for the whole `batch_audio_paths`.
            # We need to select the text_embeddings corresponding to `current_batch_actual_paths`.
            
            # Create a mapping from original index in batch_audio_paths to new index in audio_batch_tensors
            original_indices_for_loaded_audio = []
            for audio_path_loaded in current_batch_actual_paths:
                try:
                    original_indices_for_loaded_audio.append(batch_audio_paths.index(audio_path_loaded))
                except ValueError:
                    # This should not happen if current_batch_actual_paths is a subset of batch_audio_paths
                    pass 
            
            if original_indices_for_loaded_audio:
                # Select the corresponding text embeddings
                relevant_text_embeddings = batch_text_embeddings[original_indices_for_loaded_audio]

                with torch.no_grad():
                    audio_embeddings = model.get_audio_embedding_from_data(x=audio_batch_stacked, use_tensor=True)
                    audio_embeddings = audio_embeddings.to(device) 
                
                relevant_text_embeddings = relevant_text_embeddings.to(device)
                
                if audio_embeddings.shape[0] == relevant_text_embeddings.shape[0] and audio_embeddings.shape[0] > 0:
                    cosine_sim_batch = torch.nn.functional.cosine_similarity(audio_embeddings, relevant_text_embeddings, dim=1, eps=1e-8)
                    
                    # Place computed scores into the correct positions in batch_scores
                    for idx, original_batch_idx in enumerate(original_indices_for_loaded_audio):
                        batch_scores[original_batch_idx] = cosine_sim_batch[idx].cpu().item()
                else:
                    # This case means there's a mismatch, fill with NaN for safety, though handled by init.
                    # (Already initialized to NaN, so this is more for clarity)
                    print(f"Warning: Embedding shape mismatch in batch starting with {batch_audio_paths[0] if batch_audio_paths else 'N/A'}. Audio: {audio_embeddings.shape}, Text: {relevant_text_embeddings.shape}")

        all_scores.extend(batch_scores)

    return all_scores

if __name__ == "__main__":

    import pandas as pd

    csv_file_path = 'load/musiccaps-public.csv'
    df = pd.read_csv(csv_file_path)
    id2text = df.set_index('ytid')['caption'].to_dict()

    generated_path = 'your_model_outputs_folder'

    """
    IMPORTANT: the audios in generated_path should have the same ids as in id2text.
    For musiccaps, you can load id2text as above and each generated_path audio file
    corresponds to a prompt (text description) in musiccaps. Files are named with ids, as follows:
    - your_model_outputs_folder/_-kssA-FOzU.wav
    - your_model_outputs_folder/_0-2meOf9qY.wav
    - your_model_outputs_folder/_1woPC5HWSg.wav
    ...
    - your_model_outputs_folder/ZzyWbehtt0M.wav
    """

    clp = clap_score(id2text, generated_path, audio_files_extension='.wav')
    print('CLAP score (630k-audioset-fusion-best.pt): ', clp)