# app/query.py
import os
import argparse
from typing import List, Tuple, Optional, Dict
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

# -----------------------------
# 1) 検索クエリ生成（リライト）
# -----------------------------
def generate_search_queries(user_query: str, n: int = 3) -> List[str]:
    """
    ユーザー質問を、検索に適した短いクエリ（キーワード/短文）にリライトする。
    失敗時は元のクエリのみを返す。
    """
    try:
        sys_prompt = (
            "あなたは検索クエリ生成アシスタントです。"
            "与えられた質問文から、検索に適した日本語の短いクエリを最大3件、箇条書きで出力してください。"
            "同義語・別表現・関連語を混ぜてください。余計な説明はしないでください。"
        )
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_query},
            ],
            temperature=0.2,
            max_tokens=128,
        )
        lines = [
            l.strip("-・* \t")
            for l in resp.choices[0].message.content.splitlines()
            if l.strip()
        ]
        uniq = []
        for l in lines:
            if l not in uniq:
                uniq.append(l)
        queries = [user_query] + uniq
        return queries[: max(1, n + 1)]
    except Exception:
        return [user_query]

# -----------------------------
# 2) ベクトル化
# -----------------------------
def embed_many(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# -----------------------------
# 3) 検索（複数クエリ→統合）
# -----------------------------
def retrieve(
    user_query: str,
    k: int = 8,
    max_chapter_allowed: Optional[int] = None,
    use_rewrite: bool = True,
) -> Tuple[List, List[str]]:
    """
    検索クエリを生成 → 各クエリで検索 → 結果統合。
    use_rewrite=False の場合は、質問文そのままで検索。
    """
    if use_rewrite:
        queries = generate_search_queries(user_query, n=3)
    else:
        queries = [user_query]

    query_vectors = embed_many(queries)

    flt = None
    if max_chapter_allowed is not None:
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="chapter",
                    range=qmodels.Range(gte=1, lte=max_chapter_allowed),
                )
            ]
        )

    merged: Dict[str, any] = {}
    for v in query_vectors:
        hits = qdr.search(
            collection_name=COLLECTION,
            query_vector=v,
            limit=max(k * 6, 48),
            with_payload=True,
            query_filter=flt,
        )
        for h in hits:
            if (h.id not in merged) or (h.score > merged[h.id].score):
                merged[h.id] = h

    fused = sorted(merged.values(), key=lambda x: -x.score)[:k]
    return fused, queries

# -----------------------------
# 4) プロンプト構築
# -----------------------------
def build_prompt(query: str, hits) -> Tuple[str, List[Tuple[int, int, int]]]:
    ctx_blocks = []
    cites: List[Tuple[int, int, int]] = []
    for h in hits:
        p = h.payload
        text = p["text"]
        ctx_blocks.append(
            f"[chapter {p['chapter']} | {p['start_pos']}-{p['end_pos']}]\n{text}"
        )
        cites.append((p["chapter"], p["start_pos"], p["end_pos"]))
    context = "\n\n---\n\n".join(ctx_blocks)
    user = (
        f"【質問】{query}\n\n"
        f"【参照（検索で見つかった本文抜粋）】\n{context}\n\n"
        "上の参照の範囲で、簡潔かつ日本語で回答してください。"
        "参照に無い事実は『不明』と答え、推測は推測と明記してください。"
        "最後に参照箇所（chapterと位置）を列挙してください。"
    )
    return user, cites

# -----------------------------
# 5) 質問→検索→生成（本体）
# -----------------------------
def ask(query: str, max_chapter_allowed: Optional[int] = None, use_rewrite: bool = True):
    hits, generated_queries = retrieve(
        query, k=8, max_chapter_allowed=max_chapter_allowed, use_rewrite=use_rewrite
    )

    # 🔍 生成された検索クエリを表示
    print("\n[検索クエリ]")
    if use_rewrite:
        for i, q in enumerate(generated_queries, 1):
            print(f"{i}. {q}")
    else:
        print(f"(リライト無効) {generated_queries[0]}")

    user_msg, cites = build_prompt(query, hits)

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
    )
    answer = resp.choices[0].message.content
    return answer, cites

# -----------------------------
# 6) CLI（章制限 + リライトモード切替）
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG CLI with optional query rewrite.")
    parser.add_argument("--rewrite", action="store_true", help="Enable query rewrite mode")
    args = parser.parse_args()

    print(f"\n🔧 クエリ生成モード: {'ON (rewrite enabled)' if args.rewrite else 'OFF (direct query)'}")

    while True:
        q = input("\nQ> ").strip()
        if not q:
            break

        chap_str = input("max chapter allowed? (空なら全章) > ").strip()
        chap = int(chap_str) if chap_str else None

        ans, cites = ask(q, max_chapter_allowed=chap, use_rewrite=args.rewrite)

        print("\n---\n", ans)
        print("\n[参照元]")
        for c in cites:
            print(f"chapter {c[0]} ({c[1]}–{c[2]})")
