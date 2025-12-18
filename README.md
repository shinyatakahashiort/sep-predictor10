
# 小児術後屈折予測 AI アプリケーション

このリポジトリは、小児白内障手術後の屈折値（SE: Spherical Equivalent）を予測するための機械学習モデルとWebアプリケーションを含んでいます。

## ファイル構成

* `predictor.py`: 学習済みモデルをロードし、予測を行うためのクラス `SEPredictor` を定義したスクリプト。
* `streamlit_app.py`: Streamlitを使用したWebアプリケーションのメインスクリプト。
* `*_model.pkl`: 学習済みの機械学習モデル（MLP, ExtraTrees, CatBoost）。
* `*_scaler.pkl`: 特徴量スケーリング用のファイル。
* `metadata.json`: モデルの性能指標や設定情報。
* `requirements.txt`: 必要なPythonライブラリの一覧。

## 実行方法

### ローカルでの実行

1. 必要なライブラリをインストールします:
   ```bash
   pip install -r requirements.txt
   ```

2. Streamlitアプリを起動します:
   ```bash
   streamlit run streamlit_app.py
   ```

### Streamlit Cloud へのデプロイ

1. このフォルダの内容をGitHubリポジトリにアップロードします。
2. [Streamlit Cloud](https://streamlit.io/cloud) にログインします。
3. 新しいアプリを作成し、アップロードしたリポジトリを選択します。
4. "Main file path" に `streamlit_app.py` を指定してデプロイします。

## モデルについて

使用されているモデルは以下の通りです:
* **CatBoost Regressor**
* **Extra Trees Regressor**
* **MLP Regressor (Neural Network)**

最終的な予測値は、これらのモデルの予測値の加重平均（交差検証のR²スコアに基づく重み付け）によって算出されます。
