import os
import re
import json
from bs4 import BeautifulSoup
from markdownify import markdownify as md

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data", "raw_html"))
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data", "raw_text"))

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_next_data(html: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    script_tag = soup.find("script", id="__NEXT_DATA__")
    if script_tag is None or not script_tag.string:
        return None
    try:
        return json.loads(script_tag.string)
    except json.JSONDecodeError:
        return None


def build_markdown(next_data: dict) -> str | None:
    try:
        page_props = next_data["props"]["pageProps"]
        article = page_props["articleData"]
    except KeyError:
        return None

    title = (article.get("name") or "").strip()
    content_html = article.get("content") or ""
    if not content_html:
        return None

    body_md = md(content_html, heading_style="ATX", bullets="-").strip()
    body_md = re.sub(r"\n{3,}", "\n\n", body_md)  # схлопываем лишние пустые строки

    product = page_props.get("productData", {}) or {}
    section = page_props.get("sectionData", {}) or {}

    date = article.get("lastModifiedAt", "")
    slug = article.get("sefUrl", "")
    product_name = product.get("name", "")
    section_name = section.get("name", "")

    frontmatter_lines = ["---"]
    if title:
        frontmatter_lines.append(f'title: "{title}"')
    if product_name:
        frontmatter_lines.append(f'product: "{product_name}"')
    if section_name:
        frontmatter_lines.append(f'section: "{section_name}"')
    if date:
        frontmatter_lines.append(f'updated: "{date}"')
    if slug:
        frontmatter_lines.append(f'slug: "{slug}"')
    frontmatter_lines.append("---")
    frontmatter = "\n".join(frontmatter_lines)

    parts = [frontmatter]
    if title:
        parts.append(f"# {title}")
    parts.append(body_md)

    return "\n\n".join(p for p in parts if p).strip()


processed, skipped_index, fallback_raw, failed = 0, 0, [], []

for root, dirs, files in os.walk(INPUT_DIR):
    for file_name in files:
        if not file_name.endswith('.html'):
            continue

        if file_name.lower() == 'index.html':
            # страница-листинг раздела (список статей), не статья — пропускаем
            skipped_index += 1
            continue

        input_path = os.path.join(root, file_name)
        rel_path = os.path.relpath(root, INPUT_DIR)
        target_dir = os.path.normpath(os.path.join(OUTPUT_DIR, rel_path))
        os.makedirs(target_dir, exist_ok=True)
        output_path = os.path.join(target_dir, file_name.replace('.html', '.md'))

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                html = f.read()

            next_data = extract_next_data(html)
            result = build_markdown(next_data) if next_data else None

            if result is None:
                # __NEXT_DATA__ не найден/пуст на не-index файле — не теряем данные молча,
                # сохраняем через markdownify как запасной вариант
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup.select("script, style"):
                    tag.decompose()
                result = md(str(soup), heading_style="ATX", bullets="-").strip()
                fallback_raw.append(input_path)
                print(f"[fallback] __NEXT_DATA__/articleData не найден: {input_path}")

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)

            processed += 1

        except Exception as e:
            failed.append(input_path)
            print(f"Ошибка при обработке {file_name} в {root}: {e}")

print(f"\nГотово. Обработано статей: {processed}")
print(f"Пропущено index.html: {skipped_index}")
if fallback_raw:
    print(f"Через fallback (без __NEXT_DATA__), проверить вручную: {len(fallback_raw)}")
if failed:
    print(f"Ошибки: {len(failed)}")