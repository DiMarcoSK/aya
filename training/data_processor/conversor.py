import json


class Conversor:
    def process_file(self, input_file: str, output_file: str) -> None:
        data = []
        with open(input_file, encoding='utf-8') as f:
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
