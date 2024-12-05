from langchain_aws import ChatBedrock
# from langchain_openai import ChatOpenAI
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

def enhance_prompt(original_prompt, claude_enhance_params):
    input_prompt = f"""
You are a top-tier prompt engineer for Stable Diffusion. Your task is to analyze an original prompt and create variations that maintain the woman's face while altering other aspects to generate diverse female outputs.

Here is the original prompt:
<original_prompt>
{original_prompt}
</original_prompt>

First, carefully analyze the original prompt, paying attention to the descriptions of the woman's appearance, pose, setting, and any other relevant details.

Now, create a new prompt that:
1. Keeps the woman's face exactly the same as described in the original prompt
2. Modifies one or more of the following aspects:
   - Country of origin
   - Age
   - Hair style
   - Eye color
   - Skin tone
   - Body type
   - Facial expression
   - Background
   - Hair color
   - Clothing
   - Pose
   - Camera angle
3. Adds elements to make the overall image slightly sexier, without being explicit or inappropriate
4. Maintains the general style and mood of the original prompt

When crafting your new prompt:
- Use vivid, descriptive language to clearly convey the desired changes
- Ensure the modifications are cohesive and create an interesting variation
- Incorporate relevant Stable Diffusion-specific terms or techniques if appropriate

Your output should be in English and formatted as a Stable Diffusion prompt. Enclose your entire output within <output> tags.

Remember, your goal is to create an engaging variation that showcases a different type of woman while maintaining the core essence of the original prompt.
"""
    
    output = claude_invoke_model(input_prompt,{})
    return output

def generate_image_from_prompt(prompt, negative , revision_no ,model_params={}):
    # url = "http://127.0.0.1:7860"
    url = "http://192.168.68.123:7860"

    seed = random.randint(0, 4294967295)

    payload = {
        "prompt": prompt,
        "negative_prompt": negative,
        "seed": seed,
        "quality": 1.0,
        "steps": 30,
        "height": 1280,
        "width": 720,
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


    original_prompt = """
score_9, score_8_up, score_7_up, score_6_up, 
1girl, 25 years old, perfect face, (adult), sexy,
purple eyes, 
multicolored hair, colored inner hair, white hair, purple glowing hair, purple glowing scarf, 
white sweater, long sweater, long sleeves,
purple nails, 
purple skirt, layered skirt, 
white thighhighs, 
winter, snow, snowing, snowflakes,
blush, dynamic pose, dynamic angle,
night, dark
"""

    negative_prompt = """
score_6, score_5, score_4, pony, furry, monochrome, curvy, fat, pubic hair, watermark, 
artist name, ugly, ugly face, mutated hands, low res, bad anatomy, bad eyes, blurry face, unfinished, sketch, greyscale, (deformed), (child), (loli), large bimbo lips, very big eyes, (young), midriff, naked, navel, dark skin,
"""

    for i in range(30):
        prompt = enhance_prompt(original_prompt, {})
        prompt = prompt.split("<output>")[1].split("</output>")[0]
        print(f"Enhanced Prompt: {prompt}")

        start_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
   
        try:
            revision_no = start_timestamp + f"_{i + 1}"
            filename = generate_image_from_prompt(prompt,negative_prompt,revision_no ,{})

            image_path = f"SDXL/outputs/{filename}"
        except Exception as e:
            print(f"Error: {e}")
            continue

