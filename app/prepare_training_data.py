import json
from datasets import Dataset

with open("../data/bfsidata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

formatted_data = []

for item in data:
    instruction = item["instruction"]
    input_text = item.get("input", "")
    output = item["output"]

    text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""

    formatted_data.append({"text": text})

dataset = Dataset.from_list(formatted_data)
dataset.save_to_disk("../data/training_dataset")

print("Training dataset prepared successfully.")
