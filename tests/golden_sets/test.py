import json
import sys

def print_queries(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'query' in data:
                    print(data['query'])
                else:
                    print(f"[строка {line_num}] Атрибут 'query' не найден", file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"[строка {line_num}] Ошибка парсинга JSON: {e}", file=sys.stderr)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python script.py <файл.jsonl>")
        sys.exit(1)
    print_queries(sys.argv[1])