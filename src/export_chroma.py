"""Export chunks from the Day 6 Chroma store to the Day 7 chunk JSON format.

Usage:
    python src/export_chroma.py --chroma <path/to/chroma_db> --collection day6_A --output starter/data/chunks_A.json
"""
import argparse
import json
from pathlib import Path

import chromadb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--chroma', required=True, type=Path)
    ap.add_argument('--collection', default='day6_A')
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()

    col = chromadb.PersistentClient(path=str(args.chroma)).get_collection(args.collection)
    r = col.get(include=['documents', 'metadatas'])
    chunks = []
    for cid, text, meta in zip(r['ids'], r['documents'], r['metadatas']):
        meta = dict(meta)
        chunks.append({
            'id': cid,
            'text': text,
            'source_id': meta.pop('document_id'),
            'source': meta.pop('source'),
            'page': int(meta.pop('page')),
            'metadata': meta,  # chapter, section, page_id, char_len
        })
    chunks.sort(key=lambda c: (c['page'], c['id']))
    if len({c['id'] for c in chunks}) != len(chunks):
        raise ValueError('duplicate chunk ids')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{len(chunks)} chunks from {args.collection} -> {args.output}')


if __name__ == '__main__':
    main()
