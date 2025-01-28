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
    
    # llm = ChatBedrock(
    #     region_name='us-east-1',
    #     model_id='anthropic.claude-3-sonnet-20240229-v1:0',
    #     # model_id='anthropic.claude-3-haiku-20240307-v1:0',
    # )

    llm = ChatOpenAI(model="gpt-4o-mini")

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
2. Modifies all of the following aspects:
   - Clothing (e.g., style, color, fit)(note: no nudity or explicit content and no breast exposure)
   - Pose (e.g., spread legs, back arch, sitting on a chair, kneeling, lying down, etc.)
   - Camera angle
3. Adds elements to make the overall image slightly sexy, without being explicit or inappropriate
4. Add elements (looking at me from behind)
5. Add elements (T-shirts, wet:1.2)
6. Maintains the general style and mood of the original prompt

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
    url = "http://192.168.68.114:7860"
    # url = "https://4d06-60-104-55-99.ngrok-free.app/"


    seed = random.randint(0, 4294967295)

    payload = {
        "prompt": prompt,
        "negative_prompt": negative,
        "seed": seed,
        "quality": 1.0,
        "steps": 35,
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


#     original_prompt = """
# Tifa Lockhart, 1girl, detailed skin, looking at viewer, (red eyes:1.2), wearing coat and shirt, from side,
# """
    original_prompt = """
In the bathroom, a big-breasted woman with slightly wet blonde hair and blue eyes is looking at me from behind, wearing a T-shirt, Short pants, and fishnet stockings. Please put on her underwear.
"""

    negative_prompt = """
easynegative, nsfw, score_6,score_5,score_4,(worst quality:2),(low quality:1.5),(normal quality:2),bad anatomy,(artificial skin texture:1.8),poorly drawn face,poorly drawn detailed,(five-fingers),(sticky fingers:1.3),(bad finger:1.3),(bad hands:1.3),(bad fingers:1.3),(missing fingers:1.2),bad hands,poorly drawn hands,(bad fingernail:1.2),(ugly fingernail:1.2),fewer digits,source_furry,source_pony,source_cartoon,3d,blurry,(unpleasant sight),(monolids),(cross-eyed),(strabismus),(unclear eyes),(ugly eyes),(leaning eyes),(slanted eyes, upturned eyes),(unbalanced eyes),(big eyes),(small eyes),(wonky eyes),(ugly teeth:1.3),(cleft chin:1.4),(butt chin:1.4),(ugly chin),(ribs),(bad pussy:1.2),(ugly pussy:1.2),(bad anal:1.2),(ugly anal:1.2),(foundation ,makeup:1.3),(rouge, mascara:1.3),(eyelashes, false eyelashes, eyeliner:1.3),(wrinkle:1.3),watermark,logo,(claws:1.5),
"""

    for i in range(50):
        prompt = enhance_prompt(original_prompt, {})
        try:
            prompt = prompt.split("<output>")[1].split("</output>")[0]
        except:
            print(f"Error: {prompt}")
            continue

        print(f"Enhanced Prompt: {prompt}")

        start_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
   
        try:
            revision_no = start_timestamp + f"_{i + 1}"
            filename = generate_image_from_prompt(prompt,negative_prompt,revision_no ,{})

            image_path = f"SDXL/outputs/{filename}"
        except Exception as e:
            print(f"Error: {e}")
            continue

