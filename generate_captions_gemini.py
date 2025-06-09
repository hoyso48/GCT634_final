import base64
import os
import mimetypes # 파일 확장자를 통해 MIME 타입을 추론하기 위해 사용
import pandas as pd
from tqdm import tqdm
import argparse
import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt
from google import genai
# gemini_api_key.py에서 API_KEY를 불러옵니다.
from gemini_api_key import API_KEY
from google.genai.types import HttpOptions
import time
client = genai.Client(api_key=API_KEY)#, http_options=HttpOptions(timeout=60))

# 프롬프트 정의. DX7 신디사이저 오디오 콘텐츠에 대한 설명을 요청합니다.
# `{}` 부분은 이제 row_data_name으로 포맷되지 않으므로 제거했습니다.
PROMPT = "Describe the audio content: the description should start with the general type of sound (do not include the word 'synthesized'), followed by a comma (e.g., electric piano, which...), and then a general characteristic of the timbre and envelope in one sentence. For your reference, the name of the patch used to generate this sound is '{}'." #"Describe the audio content, which is generated from the DX7 synthesizer."
# 예시: "Describe the audio content: the description should start with the general type of sound (do not include the word 'synthesized'), followed by a comma (e.g., electric piano, which...), and then a general characteristic of the timbre and envelope in one sentence. For your reference, the name of the patch used to generate this sound is '{}'."
# from google.genai.errors import APIError
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(5))
async def generate_inline_content(client, audio_path, model, prompt_text):
    """
    오디오 파일을 Google 저장소에 업로드하지 않고, 인라인 데이터로 Gemini API에 전송하여 콘텐츠를 생성합니다.
    """
    
    # 1. 파일의 MIME 타입 추론
    mime_type, _ = mimetypes.guess_type(audio_path)
    if mime_type is None:
        # mimetypes 모듈이 MIME 타입을 추론하지 못할 경우를 대비한 대체 로직
        ext = os.path.splitext(audio_path)[1].lower()
        if ext == '.wav':
            mime_type = 'audio/wav'
        elif ext == '.mp3':
            mime_type = 'audio/mpeg' # MP3의 올바른 MIME 타입
        elif ext == '.m4a':
            mime_type = 'audio/mp4' # M4A의 경우 audio/mp4 사용W
        elif ext == '.aac':
            mime_type = 'audio/aac'
        else:
            # 지원하지 않는 확장자이거나 추론이 불가능한 경우 오류 발생
            raise ValueError(f"Could not determine MIME type for {audio_path}. Please ensure it's a common audio format like .wav, .mp3, .m4a.")

    # 2. 오디오 파일을 이진(binary) 모드로 읽기
    with open(audio_path, 'rb') as audio_file:
        audio_bytes = audio_file.read()

    # 3. 이진 데이터를 Base64로 인코딩
    # API는 Base64 인코딩된 문자열을 기대합니다.
    encoded_audio_data = base64.b64encode(audio_bytes).decode('utf-8')

    # 4. API 요청의 contents 필드 구성
    # 텍스트 프롬프트와 인라인 오디오 데이터를 포함합니다.
    contents = [
        prompt_text,
        {
            "inline_data": {
                "data": encoded_audio_data,
                "mime_type": mime_type
            }
        }
    ]

    # 5. Gemini API 호출
    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
    )
    return response.text

async def process_row(client, row_data, model, prompt_text):
    """
    단일 데이터 행을 처리하여 오디오 캡션을 생성합니다.
    """
    audio_path = os.path.join('data', 'wav', row_data['wav_path']) # OS 독립적인 경로 조합
    name = row_data['name']
    try:
        # generate_inline_content 함수를 호출하여 파일을 업로드하지 않고 바로 데이터 전송
        caption = await generate_inline_content(client, audio_path, model, prompt_text.format(name))
        print(f"Processed: {row_data['wav_path']}")
        print(f"Caption: {caption}")
    except Exception as e:
        print(f"Failed to process item {row_data['wav_path']} after multiple retries: {e}. Setting caption to empty.")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Root cause: {e.__cause__}")
        caption = ''

    return row_data['wav_path'], caption

async def main():
    """
    메인 함수: 스크립트 실행의 진입점
    """
    parser = argparse.ArgumentParser()
    # gemini-1.5-flash 모델은 오디오를 지원하며, 더 큰 파일 크기 제한을 가집니다.
    # 하지만 인라인 데이터 방식의 20MB 제한은 여전히 적용됩니다.
    parser.add_argument('--model', type=str, default='gemini-2.5-flash-preview-05-20') 
    parser.add_argument('--data_csv_path', type=str, default='data/DX7_YAMAHA.csv')
    parser.add_argument('--output_csv_path', type=str, default='data/DX7_YAMAHA_captions_inline.csv') # 출력 파일명 변경
    parser.add_argument('--batch_size', type=int, default=10) # 파일 크기에 따라 배치 크기를 조절해야 할 수 있습니다.
    args = parser.parse_args()

    model = args.model
    output_csv_path = args.output_csv_path
    data_csv_path = args.data_csv_path
    batch_size = args.batch_size

    df = pd.read_csv(data_csv_path)
    data_required_columns = ['id', 'wav_path', 'name'] # 'name' 컬럼이 필요하다면 추가
    for col in data_required_columns:
        if col not in df.columns:
            raise ValueError(f"Required columns {data_required_columns} not found in {data_csv_path}")
    df = df[data_required_columns]

    df_captions = df.copy()
    df_captions['caption'] = ''

    all_results = []
    rows_to_process = df_captions.to_dict('records') # DataFrame을 리스트 오브 딕셔너리로 변환

    print(f"Total rows to process: {len(rows_to_process)}")

    for i in tqdm(range(0, len(rows_to_process), batch_size), desc="Processing batches"):
        batch_rows = rows_to_process[i:i + batch_size]
        tasks = []
        for row_data in batch_rows:
            # prompt_text에 row_data['name']을 포맷하고 싶다면 여기서 처리
            # 예: current_prompt = prompt.format(row_data['name'])
            # 하지만 현재 `prompt` 변수에는 `{}` 포맷팅이 없으므로 직접 사용
            tasks.append(process_row(client, row_data, model, PROMPT)) 
        
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
        for wav_path, caption_text in batch_results:
            # df_captions에 결과 업데이트 (원본 데이터프레임과 병합하는 방식도 고려 가능)
            df_captions.loc[df_captions['wav_path'] == wav_path, 'caption'] = caption_text
            df_captions.to_csv(output_csv_path[:-4] + f'_temp.csv')
        print(f"Completed batch {i // batch_size + 1}/{(len(rows_to_process) + batch_size - 1) // batch_size}")
        time.sleep(60)

    # 결과 취합
    for wav_path, caption_text in all_results:
        # df_captions에 결과 업데이트 (원본 데이터프레임과 병합하는 방식도 고려 가능)
        df_captions.loc[df_captions['wav_path'] == wav_path, 'caption'] = caption_text
    
    # 캡션이 생성된 고유한 wav_path 개수 확인
    print(f"Number of unique captions generated: {df_captions['caption'].astype(bool).sum()}") # 비어있지 않은 캡션의 개수
    
    # 결과를 CSV 파일로 저장
    df_captions.to_csv(output_csv_path)
    print(f"Results saved to {output_csv_path}")

if __name__ == '__main__':
    asyncio.run(main())