from google import genai
import pandas as pd
from tqdm import tqdm
import argparse
import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt
from gemini_api_key import API_KEY
client = genai.Client(api_key=API_KEY)
prompt = "Describe the audio content, which is generated from the DX7 synthesizer." #"Describe the audio content: the description should start with the general type of sound (do not include the word 'synthesized'), followed by a comma (e.g., electric piano, which...), and then a general characteristic of the timbre and envelope in one sentence. For your reference, the name of the patch used to generate this sound is '{}'."

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(5))
async def upload_and_generate(client, audio_path, model, prompt, row_data_name):
    uploaded_file = await client.aio.files.upload(file=audio_path)
    # prompt = prompt.format(row_data_name)
    response = await client.aio.models.generate_content(
        model=model,
        contents=[prompt, uploaded_file]
    )
    client.files.delete(name=uploaded_file.name)
    return response.text

async def process_row(client, row_data, model, prompt):
    audio_path = 'data/wav/' + row_data['wav_path']
    caption = ''
    try:
        caption = await upload_and_generate(client, audio_path, model, prompt, row_data['name'])
        print(row_data['wav_path'])
        print(caption)
    except Exception as e:
        print(f"Failed to process item {row_data['wav_path']} after multiple retries: {e}. Setting caption to empty.")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Root cause: {e.__cause__}")
        caption = ''

    return row_data['wav_path'], caption

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gemini-2.5-flash-preview-05-20')
    parser.add_argument('--data_csv_path', type=str, default='data/DX7_YAMAHA.csv')
    parser.add_argument('--output_csv_path', type=str, default='data/DX7_YAMAHA_captions.csv')
    parser.add_argument('--batch_size', type=int, default=1000)
    args = parser.parse_args()
    model = args.model
    output_csv_path = args.output_csv_path
    data_csv_path = args.data_csv_path
    batch_size = args.batch_size

    df = pd.read_csv(data_csv_path)
    data_required_columns = ['id', 'wav_path']
    for col in data_required_columns:
        if col not in df.columns:
            raise ValueError(f"Required columns {data_required_columns} not found in {data_csv_path}")
    df = df[data_required_columns]

    df_captions = df.copy()
    df_captions['caption'] = ''

    all_results = []
    rows_to_process = []
    for i in tqdm(range(len(df_captions))):
        rows_to_process.append(df_captions.iloc[i])

    for i in tqdm(range(0, len(rows_to_process), batch_size)):
        batch_rows = rows_to_process[i:i + batch_size]
        tasks = []
        for row_data in batch_rows:
            tasks.append(process_row(client, row_data, model, prompt))
        
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
        print(f"Processed batch {i // batch_size + 1}/{(len(rows_to_process) + batch_size - 1) // batch_size}")

    for wav_path, caption_text in all_results:
        df_captions.loc[df_captions['wav_path'] == wav_path, 'caption'] = caption_text
    
    print(len(df_captions['caption'].unique()))
    df_captions.to_csv(output_csv_path)

if __name__ == '__main__':
    asyncio.run(main())