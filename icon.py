from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import HumanMessagePromptTemplate
import datetime
import random
import json
import base64
import requests

def claude_invoke_model(prompt, image_media_type=None, image_data_base64=None, model_params={}):
    
    llm = ChatBedrock(
        region_name='us-east-1',
        model_id='anthropic.claude-3-sonnet-20240229-v1:0',
        # model_id='anthropic.claude-3-haiku-20240307-v1:0',
    )

    # llm = ChatOpenAI(model="gpt-4o")

    messages = [
        # SystemMessage(content="ユーザーから与えられたプロンプトをSDXLで画像を生成するためのプロンプトに変換してください。"),
        HumanMessage(content=prompt),
    ]

    chain = llm | StrOutputParser()

    output = chain.invoke(messages)

    return output

def revise_prompt(original_prompt, claude_revise_params):
    input_prompt = f"""
Revise the following image generation prompt to optimize it for Stable Diffusion, incorporating best practices:
    {original_prompt}
    Please consider the following guidelines in your revision:
    1. prompt must be include 1 girl
    2. prompt must be include A detailed close-up of a beautiful female character for a game icon.
    3. background should be simple and not distracting.
    4. Be specific and descriptive, using vivid adjectives and clear nouns.
    5. Include details about composition, lighting, style, and mood.
    6. Mention specific artists or art styles if relevant.
    7. Use keywords like "highly detailed", "4k", "8k", or "photorealistic" if appropriate.
    8. Separate different concepts with commas.
    9. Place more important elements at the beginning of the prompt.
    10. Use weights (e.g., (keyword:1.2)) for emphasizing certain elements if necessary.
    11. If the original prompt is not in English, translate it to English. 
    Your goal is to create a clear, detailed prompt that will result in a high-quality image generation with Stable Diffusion.
    Please provide your response in the following JSON format:
    {{"revised_prompt":"<Revised Prompt>"}}
    Ensure your response can be parsed as valid JSON. Do not include any explanations, comments, or additional text outside of the JSON structure.
"""

    output = claude_invoke_model(input_prompt,{})
    return output

def enhance_prompt(original_prompt, claude_enhance_params):
    input_prompt = f"""
    あなたは画像生成のための重要な文章を作成する専門家です。与えられた短い文脈から、より詳細で視覚的な100文字程度の日本語の文章を作成することが目標です。

    以下のコンテキストが与えられます：
    <context>
    {original_prompt}
    </context>

    このコンテキストを基に、以下の手順で作業を進めてください：

    1. google play storeのカジュアルゲーム用のアイコンを作成するための文章を作成してください。
    2. 可愛らしい女性のキャラクターが、こちらをみているシーンを想像してください。
    3. アイコンには女性のキャラクターの顔のみを描いてください。
    4. 表情はコンテキストに合ったものを選んでください。
    5. 出力はjson形式を使用してください。

    まず、<inner_monologue>タグ内で、コンテキストから連想できる様々なアイデアやイメージをブレインストーミングしてください。その後、最も適切で視覚的に魅力的な要素を選び、100文字程度の日本語の文章にまとめてください。

    最終的な出力は<output>タグ内に日本語で記述してください。
"""

    output = claude_invoke_model(input_prompt,{})
    return output

def generate_image_from_prompt(prompt, revision_no ,model_params={}):

    url = "http://127.0.0.1:7860"

    seed = random.randint(0, 4294967295)

    payload = {
        "prompt": prompt,
        "negative_prompt": "easynegative, naked, paintings, sketches, bokeh, blur, (low quality:1.6), (normal quality:1.6), (worst quality:1.6), bad shadow, low res, jean, (monochrome, grayscale), polydactylism, skin spot, acnes, skin blemishes, age spot, (extra hands), (bad anatomy:1.5), brand, ((watermark:1.4)), bad feet, poorly drawn hands, poorly drawn face, mutation, Double vision in both eyes, bad eyes ratio, bad eyes size, bad eyes,deformed, bad proportions, gross proportions, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot, ((multiple arms, multiple fingers, broken fingers)), ng_deepnegative_v1_75t, (muscles), (nipple over clothes),(nipples sticking out of clothes),(Belly button on clothes),excessive abs,bad-hands-5",
        "seed": seed,
        "quality": 1.0,
        "steps": 30,
        "height": 512,
        "width": 512,
        "restore_faces": True,
    }

    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()

    dt_str = revision_no
    ## 画像を日付ディレクトリ下に保存
    ## 画像の保存先は、outputs/dt_str.png
    with open(f"outputs/{dt_str}.png", "wb") as f:
        f.write(base64.b64decode(r["images"][0]))

    print(f"Image generated successfully with seed: {seed}")
    return f"{dt_str}.png"

if __name__ == "__main__":

    original_prompt = "18歳の学生"

    for i in range(1, 6):
        try:
            prompt = enhance_prompt(original_prompt, {})
            prompt = prompt.split("<output>")[1].split("</output>")[0]
            print(f"Enhanced Prompt: {prompt}")
            prompt = revise_prompt(prompt, {})
            prompt = json.loads(prompt)["revised_prompt"]
            print(f"Revised Prompt: {prompt}")
            revision_no = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            image_file = generate_image_from_prompt(prompt, revision_no, {})
            print(f"Image file: {image_file}")
        except Exception as e:
            print(f"Error: {e}")
            continue
    
