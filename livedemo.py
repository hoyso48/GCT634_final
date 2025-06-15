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
from pydx7 import get_algorithm, write_syx_from_specs
import argparse
from safetensors import safe_open
from tqdm import tqdm
import threading


MODEL_PATH = "/workspace/GCT634_final/models/Qwen3_8B-fp8-filtered_full-tune" #args.model_path
OUTPUT_DIR = "/workspace/GCT634_final/outputs" #args.output_dir
OUTPUT_SYX = 'output_bank.syx'
NUM_BEAMS = 32
NUM_BEAM_GROUPS = 4
DIVERSITY_PENALTY = 100.0
NUM_RETURN_SEQUENCES = 4
SEED = 42

def generate_specs_for_caption(caption, model, tokenizer):
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
            do_sample = False,
            num_beams = NUM_BEAMS,
            num_beam_groups = NUM_BEAM_GROUPS, 
            diversity_penalty = DIVERSITY_PENALTY,
            num_return_sequences = NUM_RETURN_SEQUENCES,
        )
    return output

def run_generation(captions_list, model, tokenizer):
    """
    Takes a list of captions and runs the full generation and file-writing process.
    """
    if not captions_list:
        print("No captions provided for generation.")
        return

    print(f"\nStarting generation for {len(captions_list)} caption(s)...")

    model.eval()
    model.to("cuda")

    all_specs = []
    for caption in tqdm(captions_list, desc="Generating specs"):
        output = generate_specs_for_caption(caption, model, tokenizer)
        for x in output:
            x_decoded = tokenizer.decode(x, skip_special_tokens=True)
            print(x_decoded)
            specs = parse_last_specs(x_decoded)
            specs['name'] = caption.strip().replace(",", "_")[:10]
            validate_specs(specs)
            alg_num = get_algorithm(np.array(specs['modmatrix']), np.array(specs['outmatrix']))
            specs['algorithm'] = alg_num
            if alg_num == -1:
                print(f"Invalid algorithm for {specs['name']}")
            else:
                all_specs.append(specs)
            if 'fixed_freq' not in specs:
                specs['fixed_freq'] = [0,0,0,0,0,0]
            if 'has_fixed_freq' not in specs:
                specs['has_fixed_freq'] = any(specs['fixed_freq'])

    if not all_specs:
        print("No valid specs were generated.")
        return

    user_custom_defaults = {
        # 'lfo_speed': 50,
        # 'name': "PAD VOICE "
    }
    syx_path = os.path.join(OUTPUT_DIR, OUTPUT_SYX)
    write_syx_from_specs(all_specs, syx_path, user_defaults=user_custom_defaults)
    print(f"\nGeneration complete! ✨")
    print(f"Output saved to: {syx_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.random.seed(SEED)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = 768,   # Context length - can be longer, but uses more memory
        load_in_4bit = False,     # 4bit uses much less memory
        load_in_8bit = False, #PRECISION == "fp8",    # A bit more accurate, uses 2x memory
        full_finetuning = True, # We have full finetuning now!
        # token = "hf_...",      # use one if using gated models
    )

    import gradio as gr
    
    example_captions = [
        "Trumpet, which has a bright, brassy timbre and a clear, sustained envelope with a relatively quick attack and decay.",
        "A piano with a rich, full-bodied timbre with a sharp attack and a gradually fading sustain.",
        "Bell, which has a bright, metallic timbre with a sharp, percussive attack and a rapid decay.",
        "Timpani, which features a deep, resonant timbre characterized by a sharp attack and a moderately quick decay.",
        "Hi-hat, which has a bright, metallic timbre and a sharp attack with a very fast decay."
    ]
    #predefined_captions[np.random.choice(len(predefined_captions), 10, replace=False)]
    
    def process_captions_and_generate(*text_inputs):
        captions = [text for text in text_inputs if text and text.strip()]
        if not captions:
            return "Please enter at least one prompt to generate sounds."

        # Run generation in a background thread to not block the UI
        generation_thread = threading.Thread(target=run_generation, args=(captions, model, tokenizer))
        generation_thread.start()
        
        return "Prompts received! Check your console for generation progress. This window can be closed after generation starts."

    with gr.Blocks(
        theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg),
        title="Prompt Input for DX7 Sound Patch Generation"
    ) as demo:
        gr.Markdown("# 🔊 DX7 Sound Patch Generation from Text Prompts")
        gr.Markdown("Enter up to 5 text prompts to generate sounds. The generation process will start in your terminal after you click 'Generate'.")
        
        with gr.Row():
            with gr.Column(scale=2):
                text_inputs = []
                for i in range(5):
                    text_inputs.append(gr.Textbox(label=f"Prompt {i+1}", placeholder=f"Enter prompt {i+1}..."))
                
                generate_btn = gr.Button("🚀 Generate Sounds")
                output_text = gr.Label(value="", label="Status")

            with gr.Column(scale=1):
                gr.Markdown("### ✨ Example Prompts")
                example_html = "<div style='font-size: 1.1em; line-height: 1.6;'><ul>" + "".join([f"<li>{cap}</li>" for cap in example_captions]) + "</ul></div>"
                gr.HTML(example_html)

        generate_btn.click(fn=process_captions_and_generate, inputs=text_inputs, outputs=output_text)
    
    print("Launching Gradio interface... Open the URL in your browser.")
    demo.launch(share=True) # share=True is recommended for SSH access
