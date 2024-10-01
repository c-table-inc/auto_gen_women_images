# AutoGenWomenImages

AutoGenWomenImages は、先進的なAIモデルを使用して女性の高品質な画像を自動生成するパイプラインです。本プロジェクトは、LangChain を AWS Bedrock および OpenAI の GPT モデルと連携させ、Stable Diffusion (SDXL) 用のプロンプトを生成、強化、修正することで、多様で美しい画像を生成します。さらに、生成された画像の品質を評価する機能も含まれています。

## 目次

- [特徴](#特徴)
- [必要条件](#必要条件)
- [インストール](#インストール)
- [設定](#設定)
- [使用方法](#使用方法)
- [ディレクトリ構造](#ディレクトリ構造)
- [貢献](#貢献)
- [ライセンス](#ライセンス)
- [お問い合わせ](#お問い合わせ)
- [免責事項](#免責事項)

## 特徴

- **自動プロンプト生成**: 画像生成用の初期プロンプトを自動で生成します。
- **プロンプトの強化と修正**: 生成されたプロンプトを詳細化し、Stable Diffusion に最適化します。
- **画像生成**: 強化・修正されたプロンプトを使用して、SDXL API を介して画像を生成します。
- **画像品質評価**: 生成された画像の品質を特定の基準に基づいて評価します（現在はコメントアウトされています）。
- **エラーハンドリング**: パイプラインの実行中に発生するエラーを適切に処理します。

## 必要条件

プロジェクトをセットアップする前に、以下のものが必要です：

- **Python 3.8 以上**
- **LangChain ライブラリ**
  - `langchain_aws`
  - `langchain_openai`
  - `langchain_core`
- **AWS アカウント**（Bedrock へのアクセス権限が必要）
- **OpenAI API キー**
- **Stable Diffusion XL (SDXL) API** がローカルで稼働していること（例： [AUTOMATIC1111 の Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) を使用）
- **必要な Python パッケージ**: `requirements.txt` に記載

## インストール

1. **リポジトリをクローン**

   ```bash
   git clone https://github.com/yourusername/auto_gen_women_images.git
   cd auto_gen_women_images
   ```

2. **仮想環境を作成**

   依存関係を管理するために仮想環境の使用を推奨します。

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows の場合は `venv\Scripts\activate`
   ```

3. **依存関係をインストール**

   ```bash
   pip install -r requirements.txt
   ```

4. **SDXL API のセットアップ**

   Stable Diffusion XL (SDXL) API がローカルで稼働していることを確認します。以下は [AUTOMATIC1111 の Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) を使用する例です。

   ```bash
   git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
   cd stable-diffusion-webui
   python launch.py
   ```

   API が `http://127.0.0.1:7860` でアクセス可能であることを確認してください。

## 設定

1. **環境変数の設定**

   プロジェクトのルートディレクトリに `.env` ファイルを作成し、以下の設定を追加します：

   ```env
   AWS_REGION=us-east-1
   BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
   OPENAI_API_KEY=your_openai_api_key
   SDXL_API_URL=http://127.0.0.1:7860
   ```

## 使用方法

1. **スクリプトを実行**

   メインスクリプトを実行して、画像生成パイプラインを開始します。

   ```bash
   python main.py
   ```

2. **ワークフローの概要**

   - **プロンプトの作成**: 定義されたガイドラインに基づいて、画像生成用の初期プロンプトを生成します。
   - **プロンプトの強化**: 生成されたプロンプトを詳細化し、より具体的で高品質な画像生成に適したものにします。
   - **プロンプトの修正**: 強化されたプロンプトをさらに最適化します。
   - **画像生成**: 修正されたプロンプトを使用して、SDXL API を介して画像を生成します。
   - **（オプション）画像品質評価**: 特定の基準に基づいて生成された画像の品質を評価します（現在はコメントアウトされています）。

3. **出力**

   - 生成された画像は `outputs/` ディレクトリに、リビジョン番号とタイムスタンプに対応するファイル名で保存されます。
   - （オプション）品質が低い画像は `outputs/low_quality/` ディレクトリに移動されます（スクリプト内でコメントアウトされています）。

## ディレクトリ構造

```
auto_gen_women_images/
├── outputs/
│   └── low_quality/  # （オプション）品質の低い画像を保存
├── main.py
├── requirements.txt
├── README.md
└── .env
```

- **outputs/**: 生成された画像が保存されるディレクトリ。
- **outputs/low_quality/**: （オプション）品質基準を満たさない画像を保存するディレクトリ。
- **main.py**: 画像生成パイプラインのメインスクリプト。
- **requirements.txt**: Python 依存関係のリスト。
- **.env**: 環境変数の設定ファイル。

## ライセンス

このプロジェクトは [MIT License](LICENSE) の下でライセンスされています。
