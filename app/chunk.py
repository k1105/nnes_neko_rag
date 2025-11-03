# app/01_chunk.py
import re, json, math, pathlib

SRC = pathlib.Path("../data/main.txt")
OUT = pathlib.Path("../data/chunks.jsonl")

MAX_LEN = 1100
OVERLAP = 250

def split_paragraphs(text: str):
    # 連続空行で段落分割
    paras = re.split(r"\n{2,}", text.strip())
    # 行末の不必要な空白を整理
    return [re.sub(r"\s+\n", "\n", p.strip()) for p in paras if p.strip()]

def chunk_slide(paras):
    chunks = []
    buf = ""
    start = 0
    for p in paras:
        p = p if buf == "" else "\n\n" + p
        if len(buf) + len(p) <= MAX_LEN:
            if buf == "":
                start = len("".join(paras[:paras.index(p.strip())]))  # 粗いstartでもOK
            buf += p
        else:
            if buf:
                chunks.append(buf)
            # オーバーラップでつなぐ
            tail = buf[-OVERLAP:] if buf else ""
            buf = tail + p[: (MAX_LEN - len(tail))]
            # 余りをループで切る
            rest = p[(MAX_LEN - len(tail)) :]
            while rest:
                chunks.append(buf)
                tail = buf[-OVERLAP:]
                take = rest[: (MAX_LEN - len(tail))]
                buf = tail + take
                rest = rest[(MAX_LEN - len(tail)) :]
    if buf.strip():
        chunks.append(buf)
    return chunks

def main():
    text = SRC.read_text(encoding="utf-8")
    # 章もどき：全体を等分割で章ID付与（厳密な章があるならそれでOK）
    approx_chapters = max(1, math.ceil(len(text) / 8000))
    chapter_size = len(text) // approx_chapters

    paras = split_paragraphs(text)
    chunks = chunk_slide(paras)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        pos = 0
        for i, c in enumerate(chunks):
            # ざっくり開始位置で章番号を割り振る（実運用は厳密に）
            start_pos = pos
            end_pos = pos + len(c)
            chapter = min(approx_chapters - 1, start_pos // chapter_size) + 1
            rec = {
                "id": f"scene_{i:05d}",
                "chapter": chapter,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "text": c
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pos = end_pos

if __name__ == "__main__":
    main()

