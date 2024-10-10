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
    1. prompt must be include 1 girl and slightly sexy pose
    2. The woman to be output should resemble the actress specified in acctress.
    3  Please refer to the age_at_time value for the age of the woman to be output.
    4. Background should follow scene_description value
    5. Be specific and descriptive, using vivid adjectives and clear nouns.
    6. Include details about composition, lighting, style, and mood.
    7. Mention specific artists or art styles if relevant.
    8. Use keywords like "highly detailed", "4k", "8k", or "photorealistic" if appropriate.
    9. Separate different concepts with commas.
    10. Place more important elements at the beginning of the prompt.
    11. Use weights (e.g., (keyword:1.2)) for emphasizing certain elements if necessary.
    12. If the original prompt is not in English, translate it to English. 
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

    1. 与えられたコンテキストを注意深く分析し、主要な要素や雰囲気を把握してください。
    2. コンテキストから想像できる様々な情景、感情、細部を考えてください。
    3. 視覚的に豊かで、画像生成に適した表現を使用してください。
    4. 100文字程度の日本語の文章にまとめてください。
    5. 出力はjson形式を使用してください。

    良い出力の例：
    - 「夕暮れの海岸で、波の音を聴きながら砂浜を歩く若いカップル。オレンジ色の空を背景に、二人の影が長く伸びている。」
    - 「雪に覆われた静かな森の中、一匹の狐が足跡を残しながらそっと歩いている。枝から落ちる雪の結晶が、月明かりに輝いている。」

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
        "height": 960,
        "width": 540,
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

    movies_data = [
        {
            "title": "The Graduate",
            "actress": "Anne Bancroft",
            "age_at_time": 35,
            "scene_description": "高級ホテルの部屋で、ストッキングを履いた女性が、椅子に座りながら足を組んでいる。彼女の視線は挑発的で、背景には大きな窓からの夜景が広がる。"
        },
        {
            "title": "Saturday Night Fever",
            "actress": "Karen Lynn Gorney",
            "age_at_time": 32,
            "scene_description": "ディスコのフロアで、白いドレスを着た女性が鏡張りの壁を背景に、ディスコライトに照らされながら踊っている。彼女の自信に満ちた表情と、華やかなムードが漂う。"
        },
        {
            "title": "Blue Lagoon",
            "actress": "Brooke Shields",
            "age_at_time": 14,
            "scene_description": "南国の美しいビーチで、白い布を纏った若い女性が、砂浜に座って遠くの海を見つめている。背景には青い空と透き通った海が広がり、自然の美しさが際立つ。"
        },
        {
            "title": "Flashdance",
            "actress": "Jennifer Beals",
            "age_at_time": 19,
            "scene_description": "工場の作業場を背景に、タンクトップとレギンスを着た女性が、濡れた髪を振りながらダンスのポーズを決めている。汗がキラキラと輝き、エネルギッシュな魅力を醸し出している。"
        },
        {
            "title": "Top Gun",
            "actress": "Kelly McGillis",
            "age_at_time": 29,
            "scene_description": "軍用機の滑走路を背景に、革のジャケットを羽織った女性がサングラスをかけて立っている。風に髪が揺れる彼女の姿が強さとセクシーさを感じさせる。"
        },
        {
            "title": "Chinatown",
            "actress": "Faye Dunaway",
            "age_at_time": 33,
            "scene_description": "高級レストランのバーで、クラシカルな黒いドレスを着た女性が、グラスを持ちながらカウンターにもたれかかっている。背景にはムードのある照明が施され、彼女の神秘的な魅力が漂う。"
        },
        {
            "title": "9 to 5",
            "actress": "Dolly Parton",
            "age_at_time": 34,
            "scene_description": "会社のオフィスで、鮮やかなピンクのブラウスを着た女性が、デスクに腰掛けながら、にっこりと微笑んでいる。背景にはオフィスの書類やタイプライターが並び、70年代の職場風景が広がる。"
        },
        {
            "title": "Breakfast at Tiffany's",
            "actress": "Audrey Hepburn",
            "age_at_time": 32,
            "scene_description": "ニューヨークのティファニー前で、黒いドレスを着た女性が朝の光を浴びながら、カフェオレを手にして立っている。背景にはガラスのショーウィンドウが広がり、エレガントで上品な雰囲気が漂う。"
        },
        {
            "title": "Raiders of the Lost Ark",
            "actress": "Karen Allen",
            "age_at_time": 29,
            "scene_description": "古代の遺跡を背景に、冒険者風のシャツを着た女性が、夜明けの光を浴びて立っている。彼女の視線は前方に向けられ、探検の決意が感じられる。"
        },
        {
            "title": "Butch Cassidy and the Sundance Kid",
            "actress": "Katharine Ross",
            "age_at_time": 29,
            "scene_description": "牧場のフェンスに腰掛け、風に髪をなびかせながら遠くの地平線を見つめる女性。背景には広大な草原と夕日の輝きが広がり、自然と共にある彼女の自由な雰囲気が感じられる。"
        },
        {
            "title": "Grease",
            "actress": "Olivia Newton-John",
            "age_at_time": 29,
            "scene_description": "ドライブインシアターの駐車場で、レザージャケットを着た女性がクラシックカーのボンネットにもたれかかり、カメラに向かってウインクしている。背景には50年代風のネオンが輝き、レトロでセクシーな雰囲気。"
        },
        {
            "title": "Dirty Dancing",
            "actress": "Jennifer Grey",
            "age_at_time": 27,
            "scene_description": "リゾートの湖畔で、サマードレスを着た女性がダンスのポーズを決めている。背景には木々と湖の静かな風景が広がり、彼女のリズミカルな動きが鮮やかに映える。"
        },
        {
            "title": "Romancing the Stone",
            "actress": "Kathleen Turner",
            "age_at_time": 30,
            "scene_description": "ジャングルの中、探検スタイルの服装をした女性が地図を広げている。彼女の視線は遠くの山々に向けられ、冒険の緊張感とワイルドな魅力が感じられる。"
        },
        {
            "title": "Alien",
            "actress": "Sigourney Weaver",
            "age_at_time": 29,
            "scene_description": "宇宙船のコックピットで、宇宙服を脱ぎかけた女性が計器を見つめている。背景には青いライトが点滅し、彼女の鋭い目つきと緊張感が漂う。"
        },
        {
            "title": "Taxi Driver",
            "actress": "Jodie Foster",
            "age_at_time": 13,
            "scene_description": "ニューヨークの街角で、カジュアルなシャツとショートパンツを着た若い女性が、古い建物の前に立ち、遠くを見つめている。彼女の姿には少しのあどけなさと独特の雰囲気が漂う。"
        },
        {
            "title": "St. Elmo's Fire",
            "actress": "Demi Moore",
            "age_at_time": 23,
            "scene_description": "都会の夜のカフェで、タイトなドレスを着た女性がカウンターに座り、グラスを手にしている。背景には窓の外にネオンライトが広がり、都会的でセクシーな雰囲気。"
        },
        {
            "title": "Splash",
            "actress": "Daryl Hannah",
            "age_at_time": 23,
            "scene_description": "海辺で、白いビーチドレスを着た女性が波打ち際に立ち、潮風に髪をなびかせている。背景には水平線が続き、自然の美しさが際立つ。"
        },
        {
            "title": "Scarface",
            "actress": "Michelle Pfeiffer",
            "age_at_time": 25,
            "scene_description": "高級クラブのバーで、シルバーのドレスを着た女性がカウンターに座り、グラスを手にしている。背景にはネオンの明かりが反射している。"
        },
        {
            "title": "Halloween",
            "actress": "Jamie Lee Curtis",
            "age_at_time": 20,
            "scene_description": "秋の静かな住宅街で、カジュアルな服装をした女性が、手に懐中電灯を持ちながら歩いている。背景には紅葉の木々と古い家が並び、彼女の表情には警戒心が漂う。"
        },
        {
            "title": "Ferris Bueller's Day Off",
            "actress": "Mia Sara",
            "age_at_time": 18,
            "scene_description": "シカゴの高層ビルの展望台で、白いサマードレスを着た女性がガラス窓に手をつきながら、街の景色を眺めている。背景にはシカゴの美しいスカイラインが広がり、リラックスした雰囲気が漂う。"
        },
        {
            "title": "To Catch a Thief",
            "actress": "Grace Kelly",
            "age_at_time": 26,
            "scene_description": "リビエラの豪華なホテルのバルコニーで、シルクのガウンを羽織った女性が、夜空を眺めている。背景には地中海の美しい夜景が広がり、彼女の優雅な姿が際立つ。"
        },
        {
            "title": "Dr. No",
            "actress": "Ursula Andress",
            "age_at_time": 26,
            "scene_description": "白いビキニを着た女性が、海辺から上がってくるシーン。背景には青い海と白い砂浜が広がり、太陽の光が彼女を照らしている。"
        },
        {
            "title": "Body Heat",
            "actress": "Kathleen Turner",
            "age_at_time": 27,
            "scene_description": "暑い夏の夜、ベッドルームの窓辺で、薄手のシルクドレスを着た女性が風に髪をなびかせながら外を眺めている。背景には蒸し暑い夜の雰囲気が漂う。"
        },
        {
            "title": "Cat People",
            "actress": "Nastassja Kinski",
            "age_at_time": 22,
            "scene_description": "都会の屋上プールで、黒い水着を着た女性がプールサイドに座り、夜景を見つめている。背景にはニューヨークのネオンが輝き、神秘的でセクシーな雰囲気が漂う。"
        },
        {
            "title": "Some Like It Hot",
            "actress": "Marilyn Monroe",
            "age_at_time": 33,
            "scene_description": "高級ナイトクラブのステージで、煌びやかなドレスを着た女性がマイクを手にして歌っている。背景にはバンドと観客が見守り、彼女の魅力的な笑顔が輝いている。"
        },
        {
            "title": "Carrie",
            "actress": "Sissy Spacek",
            "age_at_time": 27,
            "scene_description": "高校のプロム会場で、華やかなドレスを着た女性がステージの中央に立ち、スポットライトを浴びている。背景には赤いカーテンとシャンデリアが輝き、彼女の儚げな美しさが際立つ。"
        },
        {
            "title": "Risky Business",
            "actress": "Rebecca De Mornay",
            "age_at_time": 24,
            "scene_description": "夜の電車の車内で、シンプルなドレスを着た女性がシートに腰掛け、窓の外の夜景を見つめている。背景にはニューヨークの街並みが流れ、彼女のセクシーでミステリアスな雰囲気が漂う。"
        },
        {
            "title": "Basic Instinct",
            "actress": "Sharon Stone",
            "age_at_time": 34,
            "scene_description": "モダンなリビングルームの椅子に座り、足を組み替えながら微笑む女性。背景にはモノクロームのインテリアが広がり、彼女の自信と挑発的な魅力が際立つ。"
        },
        {
            "title": "Bonnie and Clyde",
            "actress": "Faye Dunaway",
            "age_at_time": 26,
            "scene_description": "アメリカ南部の乾いた風景の中、ヴィンテージカーのボンネットにもたれかかり、スカートをひるがえす女性。背景には砂埃が舞い、彼女の視線は遠くの地平線に向かっている。"
        },
        {
            "title": "Working Girl",
            "actress": "Melanie Griffith",
            "age_at_time": 31,
            "scene_description": "高層ビルのオフィスで、タイトなビジネススーツを着た女性が窓際に立ち、ニューヨークの街並みを見下ろしている。背景には朝の光が差し込み、彼女の自信と美しさが映える。"
        },
        {
            "title": "Vertigo",
            "actress": "Kim Novak",
            "age_at_time": 25,
            "scene_description": "サンフランシスコの高台から、霧に包まれた街を見つめる女性。背景にはゴールデンゲートブリッジがぼんやりと見え、彼女の孤独感と神秘的な美しさが際立つ。"
        },
        {
            "title": "Klute",
            "actress": "Jane Fonda",
            "age_at_time": 33,
            "scene_description": "都会の薄暗いアパートの窓辺に立ち、夜景を眺める女性。背景にはネオンの明かりが映り込み、彼女の横顔には複雑な感情が漂う。"
        },
        {
            "title": "Shampoo",
            "actress": "Julie Christie",
            "age_at_time": 34,
            "scene_description": "カリフォルニアのビーチハウスで、カジュアルなシャツを着た女性がデッキチェアに座り、太陽の下でリラックスしている。背景には海と青空が広がり、リラックスした雰囲気が漂う。"
        },
        {
            "title": "Beverly Hills Cop",
            "actress": "Lisa Eilbacher",
            "age_at_time": 28,
            "scene_description": "高級なプールサイドで、白い水着を着た女性がラウンジチェアに横たわり、サングラスをかけている。背景にはパームツリーと豪邸が映り、彼女のリラックスした姿が際立つ。"
        },
        {
            "title": "The Color of Money",
            "actress": "Mary Elizabeth Mastrantonio",
            "age_at_time": 28,
            "scene_description": "ビリヤード場で、カジュアルなシャツを着た女性がビリヤードテーブルに寄りかかりながら、キューを手にしている。背景にはネオンライトが輝き、彼女の自信に満ちた表情が映える。"
        },
        {
            "title": "Giant",
            "actress": "Elizabeth Taylor",
            "age_at_time": 24,
            "scene_description": "広大なテキサスの牧場で、ロングドレスを着た女性が馬に乗り、風を感じながら地平線を見つめている。背景には広がる砂漠と青空が広がり、彼女の自由で大胆な姿が描かれる。"
        },
        {
            "title": "Grease",
            "actress": "Olivia Newton-John",
            "age_at_time": 29,
            "scene_description": "アメリカのドライブインシアターで、タイトな黒のパンツを履いた女性が、ヴィンテージカーのフードに腰掛けている。背景には映画のスクリーンと古い車が並び、彼女のカジュアルでセクシーなスタイルが映える。"
        },
        {
            "title": "Raiders of the Lost Ark",
            "actress": "Karen Allen",
            "age_at_time": 29,
            "scene_description": "エジプトの砂漠で、冒険的な服装をした女性が、古代遺跡の前に立ち、広大な風景を見渡している。背景には砂丘とピラミッドが並び、彼女の強さと魅力が際立つ。"
        },
        {
            "title": "Butch Cassidy and the Sundance Kid",
            "actress": "Katharine Ross",
            "age_at_time": 29,
            "scene_description": "西部の荒野で、古びたフェンスに寄りかかり、カウボーイハットをかぶった女性が遠くを見つめている。背景には広がる草原と青空があり、彼女の落ち着いた雰囲気が漂う。"
        },
        {
            "title": "Dirty Dancing",
            "actress": "Jennifer Grey",
            "age_at_time": 27,
            "scene_description": "夏のリゾートで、カジュアルなサマードレスを着た女性がダンスフロアで軽やかに踊っている。背景には緑豊かな森と湖が広がり、彼女の楽しげな笑顔が印象的。"
        },
        {
            "title": "Tootsie",
            "actress": "Jessica Lange",
            "age_at_time": 33,
            "scene_description": "ニューヨークのアパートで、カジュアルなニットを着た女性が窓辺に立ち、夕陽を見つめる。背景には、都市のスカイラインがオレンジ色に染まり、彼女の表情には希望と憂いが漂う。"
        },
        {
            "title": "Romancing the Stone",
            "actress": "Kathleen Turner",
            "age_at_time": 30,
            "scene_description": "ジャングルの中で、シャツをボタンを開けた女性が川沿いに立っている。背景には流れる川と木々が生い茂り、冒険心あふれる彼女の姿が映える。"
        },
        {
            "title": "Thelma & Louise",
            "actress": "Geena Davis",
            "age_at_time": 35,
            "scene_description": "砂漠のハイウェイ沿いで、タンクトップを着た女性がヴィンテージカーに寄りかかっている。背景には乾いた大地と青い空が広がり、彼女の表情には自由と挑戦の心が感じられる。"
        },
        {
            "title": "Body Heat",
            "actress": "Kathleen Turner",
            "age_at_time": 27,
            "scene_description": "暑い夏の夜、ビーチハウスのテラスで、白いドレスを纏った女性が風に吹かれながら佇んでいる。背景には波打ち際と月明かりが照らす海があり、彼女のシルエットが魅惑的に浮かび上がる。"
        },
        {
            "title": "Wild at Heart",
            "actress": "Laura Dern",
            "age_at_time": 23,
            "scene_description": "アメリカ南部の田舎道で、赤いドレスを着た女性が車のボンネットに座り、ギターを弾いている。背景には広がる青空と乾いた風景が映え、彼女の自由なエネルギーが感じられる。"
        },
        {
            "title": "Risky Business",
            "actress": "Rebecca De Mornay",
            "age_at_time": 24,
            "scene_description": "豪邸のリビングルームで、シルクのブラウスとタイトスカートを着た女性が大きな窓の前に立ち、夜景を見つめている。背景には、シカゴの高層ビルが広がり、彼女の視線には未来への期待が込められている。"
        },
        {
            "title": "Sayonara",
            "actress": "Miyoshi Umeki",
            "age_at_time": 28,
            "scene_description": "京都の伝統的な庭園で、着物を着た女性が池のほとりに立っている。背景には紅葉と灯籠が美しく配置され、彼女の静かな微笑みが風景と調和している。"
        },
        {
            "title": "American Gigolo",
            "actress": "Lauren Hutton",
            "age_at_time": 37,
            "scene_description": "カリフォルニアの高級マンションで、エレガントなシルクのガウンを着た女性がプールサイドでリラックスしている。背景には夕暮れのロサンゼルスの景色が広がり、彼女の余裕ある姿が映える。"
        },
        {
            "title": "10",
            "actress": "Bo Derek",
            "age_at_time": 23,
            "scene_description": "トロピカルなビーチで、白い水着を着た女性が砂浜に横たわり、波の音を聞きながらリラックスしている。背景にはヤシの木と青い海が広がり、彼女の解放感あふれる表情が特徴的。"
        },
        {
            "title": "Cat People",
            "actress": "Nastassja Kinski",
            "age_at_time": 21,
            "scene_description": "夜の動物園で、黒いコートを纏った女性が檻の前に立っている。背景には青い月明かりに照らされた動物たちの影が映り、彼女の神秘的な表情が浮かび上がる。"
        },
        {
            "title": "The Grifters",
            "actress": "Annette Bening",
            "age_at_time": 32,
            "scene_description": "古いラスベガスのカジノで、エレガントなドレスを着た女性がバーに腰掛けている。背景にはネオンライトが煌めき、彼女の挑発的な微笑みが人々の視線を引きつける。"
        },
        {
            "title": "Vertigo",
            "actress": "Kim Novak",
            "age_at_time": 25,
            "scene_description": "サンフランシスコの古い教会で、黒いドレスを着た女性が鐘楼を見上げている。背景には霧が立ち込め、彼女の緊張感あふれる表情が印象的に映る。"
        },
        {
            "title": "Last Tango in Paris",
            "actress": "Maria Schneider",
            "age_at_time": 20,
            "scene_description": "パリのアパートの一室で、シンプルなシャツを着た女性が床に座り込んでいる。背景にはアンティークな家具と古びた壁があり、彼女の表情には複雑な感情が映る。"
        },
        {
            "title": "Married to the Mob",
            "actress": "Michelle Pfeiffer",
            "age_at_time": 30,
            "scene_description": "ニューヨークの小さなカフェで、ヴィンテージのジャケットを羽織った女性が窓際に座り、外の景色を眺めている。背景には古い街並みが広がり、彼女の表情には少しの不安と希望が混じっている。"
        },
        {
            "title": "Baby Doll",
            "actress": "Carroll Baker",
            "age_at_time": 25,
            "scene_description": "田舎の農場のポーチで、白いサンドレスを着た女性が揺り椅子に座りながら、遠くの空を見つめている。背景には古い納屋と広がる畑があり、彼女の純粋な魅力が際立つ。"
        },
        {
            "title": "Basic Instinct",
            "actress": "Sharon Stone",
            "age_at_time": 34,
            "scene_description": "モダンなインテリアの部屋で、白いシルクのドレスを着た女性がソファに座り、足を組んでいる。背景にはシンプルで高級感のある家具が配置され、彼女の自信とミステリアスな表情が印象的。"
        },
        {
            "title": "Peggy Sue Got Married",
            "actress": "Kathleen Turner",
            "age_at_time": 32,
            "scene_description": "小さなアメリカの街並みを背景に、ヴィンテージドレスを着た女性が高校の校庭に立ち、遠くを見つめている。背景にはレンガ造りの建物と青い空が広がり、彼女の表情には過去への思いが込められている。"
        },
        {
            "title": "Shampoo",
            "actress": "Goldie Hawn",
            "age_at_time": 30,
            "scene_description": "カリフォルニアの美容院で、白いワンピースを着た女性が鏡の前に立ち、髪を整えている。背景にはカラフルな内装と観葉植物が配置され、彼女の明るく無邪気な笑顔が映える。"
        },
        {
            "title": "Moonstruck",
            "actress": "Cher",
            "age_at_time": 41,
            "scene_description": "ニューヨークの月明かりの下で、黒いドレスを纏った女性がバルコニーに立ち、夜空を見上げている。背景にはマンハッタンのビル群と煌く星が広がり、彼女の表情には静かな決意が感じられる。"
        },
        {
            "title": "Desperately Seeking Susan",
            "actress": "Madonna",
            "age_at_time": 27,
            "scene_description": "ニューヨークのダウンタウンで、レザージャケットとショートパンツを着た女性が街角に立ち、通りを歩く人々を眺めている。背景にはグラフィティアートと古いレンガの建物があり、彼女の反骨精神と自由な雰囲気が漂う。"
        }
    ]

    for movie in movies_data:
        try:
            print(f"Processing movie: {movie['title']}")

            ## 画像生成のためのプロンプトを作成
            prompt = "acctress: " + movie["actress"] + ", age_at_time: " + str(movie["age_at_time"]) + ", scene_description: " + movie["scene_description"]
            print(f"Prompt: {prompt}")

            ## プロンプトの修正
            claude_revise_params = {}
            revised_prompt = revise_prompt(prompt, claude_revise_params)
            print(f"Revised Prompt: {revised_prompt}")
            
            ## 画像生成
            revision_no = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            image_file = generate_image_from_prompt(revised_prompt, revision_no, {})
            print(f"Image file: {image_file}")
        except Exception as e:
            print(f"Error: {e}")
            continue

