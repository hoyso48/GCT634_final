import os
os.environ['HF_HOME'] = '/workspace/GCT634_final/huggingface'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from unsloth import FastLanguageModel
import torch
import numpy as np
import pandas as pd
from unsloth.chat_templates import train_on_responses_only
from unsloth import FastLanguageModel
from prompt import zeroshot_prompt
import ast
from utils import serialize_specs
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from utils import parse_last_specs, render_from_specs, validate_specs
# import IPython.display as ipdㄴ
from scipy.io import wavfile
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig
import wandb
from pydx7 import get_algorithm
import argparse

KEY_ORDER = ['algorithm', 'modmatrix', 'outmatrix', 'feedback', 'fixed_freq', 'coarse', 'fine', 'detune', 'transpose', 'ol', 'eg_rate', 'eg_level', 'sensitivity']
# KEYS_TO_REMOVE = ['name', 'has_fixed_freq'] #,'fixed_freq', 'modmatrix', 'outmatrix']
PREDICT_ALGORITHM = False #True
# PRECISION = "fp8" #"bf16"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen3-8B")
    parser.add_argument("--model_path", type=str, default="models/Qwen3-8B-fp8-filtered-responses-only")
    parser.add_argument("--keys_to_remove", type=str, default="['name', 'has_fixed_freq']")
    parser.add_argument("--precision", type=str, default="fp8")
    parser.add_argument("--responses_only", action='store_true')
    parser.add_argument("--filter_train", action='store_true')
    parser.add_argument("--filter_test", action='store_true')

    args = parser.parse_args()

    KEYS_TO_REMOVE = ast.literal_eval(args.keys_to_remove)
    PRECISION = args.precision
    RESPONSES_ONLY = args.responses_only
    MODEL_PATH = args.model_path
    MODEL_NAME = args.model
    FILTER_TRAIN = args.filter_train
    FILTER_TEST = args.filter_test

    os.environ["WANDB_PROJECT"] = "GCT634_final" 
    os.environ["WANDB_API_KEY"] = "d8bf357c20b12a02c68b6749a00946fead8c5341"
    wandb.login()
    # fourbit_models = [
    #     "unsloth/Qwen3-1.7B-unsloth-bnb-4bit", # Qwen 14B 2x faster
    #     "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    #     "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    #     "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    #     "unsloth/Qwen3-32B-unsloth-bnb-4bit",

    #     # 4bit dynamic quants for superior accuracy and low memory use
    #     "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    #     "unsloth/Phi-4",
    #     "unsloth/Llama-3.1-8B",
    #     "unsloth/Llama-3.2-3B",
    #     "unsloth/Qwen3-4B",
    #     "unsloth/Qwen3-8B"
    #     "unsloth/Qwen3-14B"
    #     "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [NEW] We support TTS models!
    # ] # More models at https://huggingface.co/unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 768,   # Context length - can be longer, but uses more memory
        load_in_4bit = False,     # 4bit uses much less memory
        load_in_8bit = PRECISION == "fp8",    # A bit more accurate, uses 2x memory
        full_finetuning = True, # We have full finetuning now!
        # token = "hf_...",      # use one if using gated models
    )
    # FastLanguageModel.from_pretrained?


    train_caption = pd.read_csv("data/DX7_YAMAHA_train_captions.csv", index_col=0)
    train_caption_add = pd.read_csv("data/DX7_AllTheWeb_train_captions.csv", index_col=0)
    test_caption = pd.read_csv("data/DX7_YAMAHA_test_captions.csv", index_col=0)
    train_data = pd.read_csv("data/DX7_YAMAHA_train.csv", index_col=0)
    train_data_add = pd.read_csv("data/DX7_AllTheWeb_train.csv", index_col=0)
    test_data = pd.read_csv("data/DX7_YAMAHA_test.csv", index_col=0)
    train_df = pd.merge(train_data, train_caption[['id', 'caption']], on='id', how='left')
    train_df_add = pd.merge(train_data_add, train_caption_add[['id', 'caption']], on='id', how='left')
    train_df = pd.concat([train_df, train_df_add])
    test_df = pd.merge(test_data, test_caption[['id', 'caption']], on='id', how='left')

    train_filter = (train_df['inaudible'] == False) & ~train_df['name'].str.contains('NULL')
    test_filter = (test_df['inaudible'] == False) & ~test_df['name'].str.contains('NULL')
    test_filter &= ~test_df['has_fixed_freqs']
    # train_filter &= ~train_df['has_fixed_freqs']
    if FILTER_TRAIN:
        train_df = train_df[train_filter]
    if FILTER_TEST:
        test_df = test_df[test_filter]
    print(len(train_df), len(test_df))


    train_dataset = Dataset.from_pandas(train_df[['id', 'caption', 'patch_data']])
    test_dataset = Dataset.from_pandas(test_df[['id', 'caption', 'patch_data']])


    def proprocess(examples, prompt = zeroshot_prompt):
        conversations = []
        for caption, patch_data in zip(examples["caption"], examples["patch_data"]):
            question = prompt.format(prompt=caption)
            specs = ast.literal_eval(patch_data)
            if PREDICT_ALGORITHM:
                algorithm = get_algorithm(np.array(specs['modmatrix']), np.array(specs['outmatrix']))
                if algorithm == -1:
                    raise ValueError(f"Algorithm not found for patch {patch_data}")
                specs['algorithm'] = algorithm
            for key in KEYS_TO_REMOVE:
                specs.pop(key, None)
            specs = {k: v for k, v in sorted(specs.items(), key=lambda item: KEY_ORDER.index(item[0]))}
            solution = '```python\n' + 'specs = ' + serialize_specs(specs) + '```\n'
            conversations.append([
                {"role" : "user",      "content" : question},
                {"role" : "assistant", "content" : solution},
            ])
        return { "conversations": conversations, }


    train_dataset = tokenizer.apply_chat_template(
        train_dataset.map(proprocess, batched = True)["conversations"],
        tokenize = False,
    )
    test_dataset = tokenizer.apply_chat_template(
        test_dataset.map(proprocess, batched = True)["conversations"],
        tokenize = False,
    )
    train_dataset = pd.Series(train_dataset)
    test_dataset = pd.Series(test_dataset)
    train_dataset.name = "text"
    test_dataset.name = "text"
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_dataset)).shuffle()
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_dataset))
    print(train_dataset[0])

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = test_dataset, # Can set up evaluation!
        args = SFTConfig(
            eval_strategy="steps",
            eval_steps = 100,
            dataset_text_field = "text",
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 1, # Use GA to mimic batch size!
            warmup_steps = 100,
            # warmup_ratio = 0.01,
            num_train_epochs = 1, # Set this for 1 full training run.
            learning_rate = 1e-4,#1e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit" if PRECISION == "fp8" else "adamw_torch_fused",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "wandb", # Use this for WandB etc
            bf16 = PRECISION == "bf16",
        ),
    )

    if RESPONSES_ONLY:
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|im_start|>user\n",
            response_part = "<|im_start|>assistant\n",
        )

    trainer.train()
    trainer.save_model(MODEL_PATH)

    caption = test_df.iloc[23]['caption']
    messages = [
        {"role" : "user", "content" : zeroshot_prompt.format(prompt=caption)}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
        enable_thinking = False, # Disable thinking
    )
    print(caption)

    output = model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        max_new_tokens = 1024, # Increase for longer outputs!
        # temperature = 1.0, #, top_p = 0.8, top_k = 20, # For non thinking
        # do_sample = False,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

    specs = parse_last_specs(tokenizer.decode(output[0], skip_special_tokens=True))
    validate_specs(specs)
    if 'fixed_freq' in KEYS_TO_REMOVE:
        specs['fixed_freq'] = [0,0,0,0,0,0]
    audio = render_from_specs(specs)
    # ipd.Audio(audio, rate=44100)
    wavfile.write('audio.wav', 44100, audio)