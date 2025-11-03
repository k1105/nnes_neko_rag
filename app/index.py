# app/02_index.py
import os, json, uuid, pathlib
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
client = OpenAI()

DATA = pathlib.Path("../data/chunks.jsonl")
COLLECTION = "neko_scenes"

qdr = QdrantClient(host="localhost", port=6333)

def embed_texts(texts):
    # バッチで埋め込み
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def get_dim():
    # 埋め込み次元を動的に取得（モデル変更に強い）
    test = client.embeddings.create(model=EMBED_MODEL, input="test").data[0].embedding
    return len(test)

def main():
    dim = get_dim()

    # --- コレクション作成（非推奨APIは使わない） ---
    if not qdr.collection_exists(COLLECTION):
        qdr.create_collection(
            collection_name=COLLECTION,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)
        )

    # --- アップサート ---
    BATCH_SIZE = 128
    batch_texts = []
    batch_payloads = []

    def flush_batch():
        if not batch_texts:
            return
        embs = embed_texts(batch_texts)

        # PointStructにして、id は UUID or int にする
        points = []
        for emb, payload in zip(embs, batch_payloads):
            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),     # ← ここが重要：UUID or int
                    vector=emb,
                    payload=payload
                )
            )

        qdr.upsert(collection_name=COLLECTION, points=points)

        batch_texts.clear()
        batch_payloads.clear()

    with DATA.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # 元の scene_xxx は payload に残す
            payload = {
                "scene_id": rec["id"],
                "chapter": rec["chapter"],
                "start_pos": rec["start_pos"],
                "end_pos": rec["end_pos"],
                "text": rec["text"]
            }
            batch_texts.append(rec["text"])
            batch_payloads.append(payload)

            if len(batch_texts) >= BATCH_SIZE:
                flush_batch()

    flush_batch()
    print("Indexed.")

if __name__ == "__main__":
    main()
