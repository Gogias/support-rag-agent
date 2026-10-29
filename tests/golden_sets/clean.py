import json

with open("end2end_eval.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for i in data:
    print(f'{i['id']} - {i['query']}')