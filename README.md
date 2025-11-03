# 🐈‍🖋 RAG Tutorial – 吾輩は猫である

このプロジェクトは、夏目澄石『吾輩は猫である』を題材にした **Retrieval-Augmented Generation (RAG)** の最小構成チュートリアルです。

小読本文を分割（チャンク化）し、ベクトル検索エンジン **Qdrant** と **OpenAI API** を利用して、意味的に近い文脈を参照しながら質問に回答します。

---

## 💻 ディレクトリ構成

```
rag-neko/
├── .env
├── data/
│   ├── main.txt             # 小読本文（UTF-8）
│   └── chunks.jsonl         # チャンク化後データ（自動生成）
├── qdrant_storage/          # Qdrant データ永続化ディレクトリ
└── app/
    ├── chunk.py             # チャンク分割
    ├── index.py             # ベクトル化 & Qdrant 登録
    ├── query.py             # 検索 & 生成（RAG コア）
    └── api.py               # FastAPI による API サーバ
```

---

## ⚙️ 事前準備

### 1. 必要ライブラリのインストール

※必要に応じて適宜仮想環境を構築

```bash
pip install -r requirements.txt
```

### 2. `.env` の作成

```
OPENAI_API_KEY=sk-xxxxxx
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

---

## 🐳 Qdrant の起動

```bash
cd app
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

確認：

```bash
curl http://localhost:6333/collections
```

---

## 🧩 ステップごとの実行

### ① チャンク分割

```bash
cd app
python chunk.py
```

生成結果： `data/chunks.jsonl`

### ② ベクトル化 & Qdrant 登録

```bash
python index.py
```

### ③ クエリ & 生成

```bash
python query.py
```

例：

```
Q> 吾輩はどんな性格？
---
 吾輩は、自己中心的でありながらも、他者を軽蔑する傾向を持つ猫です。特に、他の猫や人間に対して優越感を抱き、自分の知識や経験を誇示しようとします。また、他者の無知を試すような態度も見られます。さらに、主人や周囲の人々に対しても批判的で、彼らの行動や性格を観察し、時には軽蔑することもあります。

参照箇所：
- chapter 2 | 8800-9900
- chapter 1 | 2200-3300
- chapter 3 | 15956-17056
```

`--rewrite`フラグを立てて実行することで、ユーザに訊かれた質問を内部で一度リライトし、それをクエリとして実行するようにできます。

### ④ API 化（FastAPI サーバ）

```bash
python -m uvicorn app.api:app --reload --port 8000
```

ブラウザで [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) を開く。

---

## 🌐 API 呼び出し例

### curl

```bash
curl -X POST http://127.0.0.1:8000/query \
     -H "Content-Type: application/json" \
     -d '{"q":"吾輩はどんな性格？","max_chapter_allowed":1}'
```

### Python

```python
import requests

res = requests.post(
    "http://127.0.0.1:8000/query",
    json={"q": "吾輩はどんな性格？", "max_chapter_allowed": 1}
)
print(res.json()["answer"])
```

---

## 🚀 起動順序まとめ

1️⃣ **Qdrant 起動（Docker）**

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

2️⃣ **FastAPI 起動**

```bash
python -m uvicorn app.api:app --reload --port 8000
```

---

## 🧹 よくあるトラブル

| 症状                                            | 対処                                      |
| ----------------------------------------------- | ----------------------------------------- |
| `Cannot connect to Docker daemon`               | Docker Desktop を起動して再試行           |
| `Unexpected Response: 400 ... invalid point ID` | index.py で UUID を使用                   |
| `ModuleNotFoundError: No module named 'dotenv'` | `pip install python-dotenv`               |
| `.env が読めない`                               | `.env` をプロジェクトルートに置く         |
| Qdrant のデータが消えた                         | `-v` のパスが異なる。絶対パス指定で再実行 |

---

## 🧠 RAG アーキテクチャ概要

```
質問 → 埋め込み生成 → Qdrant検索 → コンテキスト構築 → GPT生成 → 回答
```

| 機能         | ファイル   | 役割                                         |
| ------------ | ---------- | -------------------------------------------- |
| チャンク分割 | `chunk.py` | 本文を意味単位で分割                         |
| ベクトル登録 | `index.py` | チャンクをベクトル化して保存                 |
| 検索 & 生成  | `query.py` | 質問に応じて検索・生成                       |
| API サーバ   | `api.py`   | 外部から呼び出すための HTTP インターフェース |

---

## 💡 次のステップ

- `max_chapter_allowed` で **ネタバレ防止** 制御を強化
- 会話履歴を保持して **連続対話型** に拡張
- Next.js などで **チャット UI** を構築

---

📘 **作成者**: Kanata Yamagishi
📅 **最終更新日**: 2025-11-03
