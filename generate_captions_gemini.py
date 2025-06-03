from google import genai
import pandas as pd
from tqdm import tqdm
import argparse
import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt
from gemini_api_key import API_KEY

client = genai.Client(api_key=API_KEY)
prompt = "Describe the audio content: the description should start with the general type of sound (do not include the word 'synthesized'), followed by a comma (e.g., electric piano, which...), and then a general characteristic of the timbre and envelope in one sentence. For your reference, the name of the patch used to generate this sound is '{}'."

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(5))
async def upload_and_generate(client, audio_path, model, prompt, row_data_name):
    uploaded_file = await client.aio.files.upload(file=audio_path)
    response = await client.aio.models.generate_content(
        model=model,
        contents=[prompt.format(row_data_name), uploaded_file]
    )
    return response.text

async def process_row(client, row_data, model, prompt, column_name):
    audio_path = 'data/wav/' + row_data['wav_path']
    caption = ''
    try:
        caption = await upload_and_generate(client, audio_path, model, prompt, row_data['name'])
        print(row_data['wav_path'])
        print(caption)
    except Exception as e:
        print(f"Failed to process item {row_data['wav_path']} after multiple retries: {e}. Setting caption to empty.")
        caption = ''

    return row_data['wav_path'], caption

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gemini-2.5-flash-preview-05-20')
    parser.add_argument('--caption_csv_path', type=str, default='data/DX7_YAMAHA_deduplicated_captions.csv')
    parser.add_argument('--data_csv_path', type=str, default='data/DX7_YAMAHA_deduplicated.csv')
    parser.add_argument('--split', type=str, default=['train', 'test'])
    parser.add_argument('--column_name', type=str, default='caption')
    parser.add_argument('--batch_size', type=int, default=1000)
    args = parser.parse_args()
    model = args.model
    caption_csv_path = args.caption_csv_path
    data_csv_path = args.data_csv_path
    split = args.split
    column_name = args.column_name
    batch_size = args.batch_size

    df = pd.read_csv(data_csv_path)
    df_captions = df[['id', 'wav_path', 'file_path', 'name', 'split']].copy()
    df_captions[column_name] = ''

    all_results = []
    rows_to_process = []
    for i in tqdm(range(len(df_captions))):
        if df_captions.iloc[i]['split'] not in split:
            continue
        rows_to_process.append(df_captions.iloc[i])

    for i in tqdm(range(0, len(rows_to_process), batch_size)):
        batch_rows = rows_to_process[i:i + batch_size]
        tasks = []
        for row_data in batch_rows:
            tasks.append(process_row(client, row_data, model, prompt, column_name))
        
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
        print(f"Processed batch {i // batch_size + 1}/{(len(rows_to_process) + batch_size - 1) // batch_size}")

    for wav_path, caption_text in all_results:
        df_captions.loc[df_captions['wav_path'] == wav_path, column_name] = caption_text
    
    print(len(df_captions[column_name].unique()))
    df_captions.to_csv(caption_csv_path, index=False)

if __name__ == '__main__':
    asyncio.run(main())