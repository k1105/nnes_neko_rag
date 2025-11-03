# app/03_query.py
import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

load_dotenv()
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

qdr = QdrantClient(host="localhost", port=6333)
client = OpenAI()
COLLECTION = "neko_scenes"

SYSTEM = """あなたは小説「吾輩は猫である」の内容に厳密に基づいて答えるアシスタントです。
不明な場合は「まだわからない」と答え、必ず引用の章と位置を最後に列挙してください。脚色や推測は明示してください。"""

def embed(q: str):
    return client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding

def retrieve(query: str, k=8, max_chapter_allowed=None):
    v = embed(query)
    # フィルタで未来章を除外（ネタバレ防止の原型）
    flt = None
    if max_chapter_allowed is not None:
        flt = qmodels.Filter(
            must=[qmodels.FieldCondition(key="chapter", range=qmodels.Range(gte=1, lte=max_chapter_allowed))]
        )
    res = qdr.search(
        collection_name=COLLECTION,
        query_vector=v,
        limit=k,
        with_payload=True,
        query_filter=flt
    )
    return res

def build_prompt(query: str, hits):
    ctx_blocks = []
    cites = []
    for h in hits:
        p = h.payload
        text = p["text"]
        ctx_blocks.append(f"[chapter {p['chapter']} | {p['start_pos']}-{p['end_pos']}]\n{text}")
        cites.append((p["chapter"], p["start_pos"], p["end_pos"]))
    context = "\n\n---\n\n".join(ctx_blocks)
    user = f"【質問】{query}\n\n【参照】\n{context}\n\n上の参照の範囲で簡潔に日本語で回答し、最後に参照箇所（chapterと位置）を列挙してください。"
    return user, cites

def ask(query: str, max_chapter_allowed=None):
    hits = retrieve(query, k=8, max_chapter_allowed=max_chapter_allowed)
    user, cites = build_prompt(query, hits)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":user}],
        temperature=0.3
    )
    answer = resp.choices[0].message.content
    return answer, cites

if __name__ == "__main__":
    while True:
        q = input("\nQ> ").strip()
        if not q: break
        # 例：いま第3章までしか読んでいない場合は max_chapter_allowed=3
        ans, cites = ask(q, max_chapter_allowed=None)
        print("\n---\n", ans)

