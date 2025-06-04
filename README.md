# GCT634 Final Project: DX7 Sound Generation and Captioning

This project involves generating audio from DX7 synthesizer patch data and then creating descriptive captions for the generated audio using a generative AI model.

## Setup

1.  **API Key**: You need to provide your Gemini API key. Create a file named `gemini_api_key.py` in the root of this project directory (`projects/GCT634_final/`) with the following content:
    ```python
    API_KEY = "YOUR_GEMINI_API_KEY"
    ```
    Replace `"YOUR_GEMINI_API_KEY"` with your actual API key.

2.  **Dependencies**: Install the required Python packages using:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Dataset Generation**: The `.csv` files used in this project (e.g., `data/DX7_YAMAHA_deduplicated.csv`) were originally generated using the `dataset.ipynb` notebook. You can refer to this notebook to understand the data preparation process.

## Usage

Below are example commands to run the different scripts in this project.

**1. Generate WAV files from Patch Data:**
```bash
python patch_to_wav.py --sr 48000 --n 64 --v 100 --out_scale 1.0 --csv_path data/DX7_YAMAHA_test.csv
```

**2. Generate Captions for Audio Files (Ground Truth):**
```bash
python generate_captions_gemini.py --data_csv_path data/DX7_YAMAHA_test.csv --model gemini-2.5-flash-preview-05-20 --output_csv_path data/DX7_YAMAHA_test_captions.csv --batch_size 1000
```

**3. Evaluate Ground Truth Audio with Captions:**
```bash
python evaluate.py --data_csv_path data/DX7_YAMAHA_test.csv --caption_csv_path data/DX7_YAMAHA_captions_test.csv --output_csv_path data/DX7_YAMAHA_scores_test_gt.csv
```

**4. Generate DX7 Specs from Captions (Gemini) and Render Audio:**
```bash
python generate_specs_gemini.py --caption_csv_path data/DX7_YAMAHA_captions_test.csv --output_csv_path data/DX7_YAMAHA_test_gemini.csv --wav_dir data/generated/gemini_test --print_response
```

**5. Evaluate Gemini-Generated Audio:**
```bash
python evaluate.py --data_csv_path data/DX7_YAMAHA_test_gemini.csv --caption_csv_path data/DX7_YAMAHA_captions_test.csv --output_csv_path data/DX7_YAMAHA_scores_test_gemini.csv --wav_dir data/generated/gemini_test
```

**6. Generate DX7 Specs from Captions (Transformers - Qwen) and Render Audio:**
```bash
python generate_specs_transformers.py --caption_csv_path data/DX7_YAMAHA_captions_test.csv --output_csv_path data/DX7_YAMAHA_test_qwen.csv --wav_dir data/generated/qwen_test --print_response
```