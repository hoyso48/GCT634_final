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


def clap_score_from_audio_text(
    audio_data_list,  # List of uint8 np.ndarray
    text_captions_list, # List of str
    batch_size=32,    # Configurable batch size for inference
    sample_rate=48000,  # Single int (input sample rate for all audios)
    clap_model_name='630k-audioset-fusion-best.pt',
    device='cuda'     # Allow device selection, e.g., 'cuda' or 'cpu'
):
    """
    Computes CLAP score from raw audio data and text captions.

    Params:
    -- audio_data_list: List of uint8 NumPy arrays representing audio waveforms.
    -- text_captions_list: List of strings (prompts) corresponding to each audio.
    -- batch_size: Batch size for processing audio and text embeddings.
    -- sample_rate: Sample rate of the input audio(s). Assumed to be the same for all audios.
    -- clap_model_name: Name of the CLAP model to use (e.g., '630k-audioset-fusion-best.pt').
    -- device: The device to run the model on ('cuda' or 'cpu').

    Returns:
    -- Average CLAP score for the provided audio-text pairs.
    """
    if not audio_data_list or not text_captions_list:
        print("Warning: Empty audio or text list provided for clap_score_from_audio_data.")
        return 0.0

    if len(audio_data_list) != len(text_captions_list):
        raise ValueError("Mismatch between number of audio samples and text captions.")

    if not isinstance(sample_rate, int):
        raise ValueError("sample_rate must be an integer.")

    # --- Model Loading ---
    model_config = {
        'music_speech_audioset_epoch_15_esc_89.98.pt': {
            'url': 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt',
            'path': 'load/clap_score/music_speech_audioset_epoch_15_esc_89.98.pt',
            'enable_fusion': False, 'amodel': 'HTSAT-base'
        },
        'music_audioset_epoch_15_esc_90.14.pt': {
            'url': 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt',
            'path': 'load/clap_score/music_audioset_epoch_15_esc_90.14.pt',
            'enable_fusion': False, 'amodel': 'HTSAT-base'
        },
        'music_speech_epoch_15_esc_89.25.pt': {
            'url': 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt',
            'path': 'load/clap_score/music_speech_epoch_15_esc_89.25.pt',
            'enable_fusion': False, 'amodel': 'HTSAT-base'
        },
        '630k-audioset-fusion-best.pt': {
            'url': 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt',
            'path': 'load/clap_score/630k-audioset-fusion-best.pt',
            'enable_fusion': True, 'amodel': None # amodel determined by fusion type
        }
    }

    if clap_model_name not in model_config:
        raise ValueError(f"clap_model_name '{clap_model_name}' not implemented. Choose from {list(model_config.keys())}")

    config = model_config[clap_model_name]
    clap_path = config['path']

    if not os.path.exists(clap_path):
        print(f'Downloading {clap_model_name} to {clap_path}...')
        os.makedirs(os.path.dirname(clap_path), exist_ok=True)
        response = requests.get(config['url'], stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(clap_path, 'wb') as file, tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {clap_model_name}") as progress_bar:
            for data in response.iter_content(chunk_size=8192):
                file.write(data)
                progress_bar.update(len(data))

    model_args = {'enable_fusion': config['enable_fusion'], 'device': device}
    if 'amodel' in config and config['amodel']: 
        model_args['amodel'] = config['amodel']
    
    model = laion_clap.CLAP_Module(**model_args)
    
    pkg = load_state_dict(clap_path, map_location='cpu') # Load to CPU first
    pkg.pop('text_branch.embeddings.position_ids', None) # Fix for CLAP issue
    model.model.load_state_dict(pkg)
    model.eval()
    model.to(device) # Move model to target device
    # --- End Model Loading ---

    # --- Get Text Embeddings ---
    # print('[EXTRACTING TEXT EMBEDDINGS FOR clap_score_from_audio_data]') # Optional: for verbose logging
    with torch.no_grad():
        all_text_embeddings = model.get_text_embedding(text_captions_list, use_tensor=True).to(device)

    # --- Process Audio and Calculate Scores in Batches ---
    # print('[EVALUATING AUDIO DATA FOR clap_score_from_audio_data]') # Optional: for verbose logging
    total_score = 0.0
    num_evaluated_samples = 0

    for i in tqdm(range(0, len(audio_data_list), batch_size), desc="Processing audio batches"):
        batch_audio_data_uint8 = audio_data_list[i : i + batch_size]
        # batch_srs_list = srs_list[i : i + batch_size] # No longer needed
        # Corresponding text embeddings for the current batch
        batch_text_embeddings = all_text_embeddings[i : i + batch_size]

        processed_audio_tensors_for_clap_model = []
        for idx_in_batch, raw_audio_uint8 in enumerate(batch_audio_data_uint8):

            if not isinstance(raw_audio_uint8, np.ndarray) or raw_audio_uint8.dtype != np.uint8:
                 raise ValueError(f"Audio data at index {i+idx_in_batch} must be a NumPy array of dtype uint8. Found: {type(raw_audio_uint8)}, dtype: {raw_audio_uint8.dtype if isinstance(raw_audio_uint8, np.ndarray) else 'N/A'}")

            # 1. Convert uint8 to float32 (range -1.0 to 1.0)
            audio_f32 = librosa.util.buf_to_float(raw_audio_uint8, n_bytes=1) # n_bytes=1 for uint8

            # 2. Resample if necessary to 48000 Hz
            audio_resampled = audio_f32
            if sample_rate != 48000:
                audio_resampled = librosa.resample(audio_f32, orig_sr=sample_rate, target_sr=48000, res_type='kaiser_best')
            
            # 3. Normalize peak to -1.0
            audio_normalized = pyln.normalize.peak(audio_resampled, -1.0)

            # 4. Apply CLAP's specific processing (mimicking int16 conversion)
            audio_final_np = int16_to_float32(float32_to_int16(audio_normalized))
            
            # Reshape for CLAP: (1, T) as expected by get_audio_embedding_from_data when passing a list
            audio_final_np_reshaped = audio_final_np.reshape(1, -1)
            processed_audio_tensors_for_clap_model.append(torch.from_numpy(audio_final_np_reshaped).float())

        # Move list of processed audio tensors to the target device
        batch_audio_tensors_on_device = [t.to(device) for t in processed_audio_tensors_for_clap_model]

        with torch.no_grad():
            # Get audio embeddings for the batch.
            # x should be a list of tensors, each tensor being one audio sample (1, T)
            current_audio_embeddings = model.get_audio_embedding_from_data(x=batch_audio_tensors_on_device, use_tensor=True)

        # Calculate cosine similarity for the batch
        cosine_sims = torch.nn.functional.cosine_similarity(current_audio_embeddings, batch_text_embeddings, dim=1, eps=1e-8)
        
        total_score += torch.sum(cosine_sims).item()
        num_evaluated_samples += len(batch_audio_data_uint8)

    return total_score / num_evaluated_samples if num_evaluated_samples > 0 else 0.0


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