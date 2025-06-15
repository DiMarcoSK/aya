import json

input_file = 'data/200_leaks.jsonl'
output_file = 'data/dataset_alpaca.json'

data = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        example = json.loads(line)
        data.append({
            "instruction": example["prompt"],
            "input": "",
            "output": example["response"]
        })

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"{output_file}")
