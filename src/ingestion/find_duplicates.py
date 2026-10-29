"""Диагностика дублей: на уровне файлов и на уровне чанков. Ничего не эмбеддит."""

from collections import defaultdict

from chunk_and_embed import (
    DATA_ROOT, build_source_path, collect_md_files, content_hash,
    file_to_chunks, parse_frontmatter,
)

files_by_hash = defaultdict(list)
chunks_by_hash = defaultdict(list)
total_chunks = 0

for p in collect_md_files(DATA_ROOT):
    meta, body = parse_frontmatter(p.read_text(encoding="utf-8"))
    src = build_source_path(p, DATA_ROOT)
    files_by_hash[content_hash(body)].append(src)
    for c in file_to_chunks(body, meta, src):
        total_chunks += 1
        chunks_by_hash[content_hash(c["content"])].append(c)

dup_files = [v for v in files_by_hash.values() if len(v) > 1]
print(f"Файлов-дублей (одинаковое тело): {sum(len(v) - 1 for v in dup_files)}")
for group in dup_files[:5]:
    print("  ", group)

dup_groups = sorted((v for v in chunks_by_hash.values() if len(v) > 1), key=len, reverse=True)
print(f"\nЧанков всего: {total_chunks}, уникальных: {len(chunks_by_hash)}, "
      f"лишних копий: {total_chunks - len(chunks_by_hash)}")
print("\nСамые частые повторы:")
for group in dup_groups[:15]:
    text = " ".join(group[0]["content"].split())[:100]
    print(f"  x{len(group):<4} {text}")
