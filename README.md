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

### 1. Generating WAV files from Patch Data

The `patch_to_wav.py` script converts DX7 patch data (typically stored in a `.csv` file) into WAV audio files.

**Example:**
```bash
python patch_to_wav.py --sr 48000 --n 64 --v 100 --out_scale 1.0 --csv_path data/DX7_YAMAHA_deduplicated.csv
```
This command will:
*   Use a sample rate (`--sr`) of 48000 Hz.
*   Generate MIDI note number (`--n`) 64 (E4).
*   Use a MIDI velocity (`--v`) of 100.
*   Apply an output scaling factor (`--out_scale`) of 1.0.
*   Read patch data from `data/DX7_YAMAHA_deduplicated.csv` and save the output WAV files to the `data/wav/` directory (by default, inferred from the CSV path).

### 2. Generating Captions for Audio Files

The `generate_captions_gemini.py` script uses the Gemini API to generate descriptive captions for the audio files.

**Example:**
```bash
python generate_captions_gemini.py --data_csv_path data/DX7_YAMAHA_deduplicated.csv --model gemini-2.5-flash-preview-05-20 --column_name caption --caption_csv_path data/DX7_YAMAHA_deduplicated_captions.csv
```
This command will:
*   Read metadata from `data/DX7_YAMAHA_deduplicated.csv`.
*   Use the `gemini-2.5-flash-preview-05-20` model for caption generation.
*   Store the generated captions in a column named `caption`.
*   Save the results (including the new captions) to `data/DX7_YAMAHA_deduplicated_captions.csv`.

## Utilities (`utils.py`)

The `utils.py` file contains helper functions. Notably, the `render_from_specs` function is crucial for converting patch parameter specifications into an audio waveform.

If you have patch data stored as a string representation of a dictionary in a `.csv` file (e.g., in a column named 'specs'), you can convert it to an audio signal like this:

```python
# Assuming 'specs_string' is the string from the CSV
# and 'utils.py' is in your Python path or same directory.
import numpy as np
from utils import render_from_specs # Make sure pydx7 is also available

specs_dict = eval(specs_string) # Caution: eval can be risky with untrusted input
audio_waveform = render_from_specs(specs_dict, sr=48000, n=60, v=100)
# audio_waveform will be a NumPy array of int16 audio data
```
The `render_from_specs` function takes the patch dictionary, sample rate (`sr`), MIDI note number (`n`), velocity (`v`), and an optional output scaling factor.