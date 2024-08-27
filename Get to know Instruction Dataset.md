# Get to know Instruction Dataset

## Use the Hugging Face datasets library to load and save a dataset.

```python
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca")
dataset.save_to_disk("Dataset")
```

* Import the load_dataset function from the datasets library.
* Load the "alpaca" dataset from the "tatsu-lab" repository on the [Hugging Face Hub](https://huggingface.co/datasets/tatsu-lab/alpaca). The alpaca dataset is a collection of instruction-following data used for training language models.
* Save the loaded dataset to a local directory named "Dataset".

|File|Description|
|-|-|
|Dataset\dataset_dict.json|It contains metadata about the dataset, including information about its structure and the different splits (if any).|
|Dataset\train|It is a directory that contains the training split of the dataset. The Alpaca dataset appears to have only a training split.|
|Dataset\train\data-00000-of-00001.arrow|It is an Arrow file containing the actual data of the training split. Arrow is a columnar memory format designed for efficient data processing and interchange.|
|Dataset\train\dataset_info.json|It contains detailed information about the dataset, such as its description, citation, and features.|
|Dataset\train\state.json|It keeps track of the state of the dataset, which can be useful for caching and version control.|

## Convert Dataset to a single Instruction-Output (Prompt-Completion) format JSON file

```python
import json
from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("D:/cmp/GetDataSet/Dataset")

# Function to convert a single example to alpaca without input format
def to_alpaca_no_input_format(example):
    return {
        "instruction": example["instruction"],
        "output": example["output"]
    }

# Convert the entire dataset to alpaca without input format
_alpaca_no_inpu_data = [to_alpaca_no_input_format(example) for example in dataset["train"]]

# Save as a single JSON file
with open("alpaca_no_input.json", "w", encoding="utf-8") as f:
    json.dump(_alpaca_no_inpu_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(_alpaca_no_inpu_data)} examples to alpaca_no_input.json")

# Save the original dataset to another JSON file
output_file_original = "original_dataset.json"
with open(output_file_original, "w", encoding="utf-8") as f:
    original_data = [example for example in dataset["train"]]
    json.dump(original_data, f, ensure_ascii=False, indent=2)

print(f"Saved original dataset to {output_file_original}")
```

* When using or referring to this format, it's often helpful to briefly describe its structure (i.e., a JSON list of objects, each with "instruction" and "output" fields) to avoid any potential confusion.

|Useage example|Description|
|-|-|
|OpenAI's GPT models|They use a similar format for their fine-tuning datasets, often referred to as "prompt-completion pairs".|
|InstructGPT|This was one of the early adopters of instruction-tuning, using a similar format.|
|Stanford Alpaca|The original Alpaca dataset used a format very close to "Instruction-Output", though it included an additional "input" field (which could be empty).|
|FLAN (Fine-tuned Language Net)|While FLAN uses a more complex format internally, many of its tasks can be represented in this instruction-output format.|

## Instruction-Output (Prompt-Completion) format JSON file example

```plaintext
[
  {
    "instruction": "Give three tips for staying healthy.",
    "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
  },
  {
    "instruction": "What are the three primary colors?",
    "output": "The three primary colors are red, blue, and yellow."
  },
  {
    "instruction": "Describe the structure of an atom.",
    "output": "An atom is made up of a nucleus, which contains protons and neutrons, surrounded by electrons that travel in orbits around the nucleus. The protons and neutrons have a positive charge, while the electrons have a negative charge, resulting in an overall neutral atom. The number of each particle determines the atomic number and the type of atom."
  },
  {
    "instruction": "How can we reduce air pollution?",
    "output": "There are a number of ways to reduce air pollution, such as shifting to renewable energy sources, encouraging the use of public transportation, prohibiting the burning of fossil fuels, implementing policies to reduce emissions from industrial sources, and implementing vehicle emissions standards. Additionally, individuals can do their part to reduce air pollution by reducing car use, avoiding burning materials such as wood, and changing to energy efficient appliances."
  },
  {
    "instruction": "Describe a time when you had to make a difficult decision.",
    "output": "I had to make a difficult decision when I was working as a project manager at a construction company. I was in charge of a project that needed to be completed by a certain date in order to meet the client’s expectations. However, due to unexpected delays, we were not able to meet the deadline and so I had to make a difficult decision. I decided to extend the deadline, but I had to stretch the team’s resources even further and increase the budget. Although it was a risky decision, I ultimately decided to go ahead with it to ensure that the project was completed on time and that the client’s expectations were met. The project was eventually successfully completed and this was seen as a testament to my leadership and decision-making abilities."
  },
]
```

## Fine-tune sample code

```python
import torch
from datasets import load_from_disk
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load the dataset
dataset = load_from_disk("D:/cmp/GetDataSet/Dataset")

# Initialize the tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Prepare the dataset
class FlexibleAlpacaDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Check if 'input' field exists and use it if present
        input_text = item.get('input', '')
        prompt = f"Instruction: {item['instruction']}\nInput: {input_text}\nOutput: {item['output']}"
        encoding = self.tokenizer(prompt, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten(),
        }

train_dataset = FlexibleAlpacaDataset(dataset["train"], tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

# Example of using the fine-tuned model
def generate_response(instruction, input_text=""):
    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test the model
print(generate_response("Give three tips for staying healthy."))
```