# Get to know GGUF Files

GGUF stands for GPT-Generated Unified Format, which is a file format used for storing large language models, particularly those based on the GPT (Generative Pre-trained Transformer) architecture.

## [Struture](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

```python
enum ggml_type: uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_COUNT,
};

enum gguf_metadata_value_type: uint32_t {
    // The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

// A string in GGUF.
struct gguf_string_t {
    // The length of the string, in bytes.
    uint64_t len;
    // The string as a UTF-8 non-null-terminated string.
    char string[len];
};

union gguf_metadata_value_t {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    bool bool_;
    gguf_string_t string;
    struct {
        // Any value type is valid, including arrays.
        gguf_metadata_value_type type;
        // Number of elements, not bytes
        uint64_t len;
        // The array of values.
        gguf_metadata_value_t array[len];
    } array;
};

struct gguf_metadata_kv_t {
    // The key of the metadata. It is a standard GGUF string, with the following caveats:
    // - It must be a valid ASCII string.
    // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
    // - It must be at most 2^16-1/65535 bytes long.
    // Any keys that do not follow these rules are invalid.
    gguf_string_t key;

    // The type of the value.
    // Must be one of the `gguf_metadata_value_type` values.
    gguf_metadata_value_type value_type;
    // The value.
    gguf_metadata_value_t value;
};

struct gguf_header_t {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    uint32_t magic;
    // The version of the format implemented.
    // Must be `3` for version described in this spec, which introduces big-endian support.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    uint32_t version;
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    uint64_t tensor_count;
    // The number of metadata key-value pairs.
    uint64_t metadata_kv_count;
    // The metadata key-value pairs.
    gguf_metadata_kv_t metadata_kv[metadata_kv_count];
};

uint64_t align_offset(uint64_t offset) {
    return offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;
}

struct gguf_tensor_info_t {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    gguf_string_t name;
    // The number of dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    uint32_t n_dimensions;
    // The dimensions of the tensor.
    uint64_t dimensions[n_dimensions];
    // The type of the tensor.
    ggml_type type;
    // The offset of the tensor's data in this file in bytes.
    //
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    //
    // Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.
    uint64_t offset;
};

struct gguf_file_t {
    // The header of the file.
    gguf_header_t header;

    // Tensor infos, which can be used to locate the tensor data.
    gguf_tensor_info_t tensor_infos[header.tensor_count];

    // Padding to the nearest multiple of `ALIGNMENT`.
    //
    // That is, if `sizeof(header) + sizeof(tensor_infos)` is not a multiple of `ALIGNMENT`,
    // this padding is added to make it so.
    //
    // This can be calculated as `align_offset(position) - position`, where `position` is
    // the position of the end of `tensor_infos` (i.e. `sizeof(header) + sizeof(tensor_infos)`).
    uint8_t _padding[];

    // Tensor data.
    //
    // This is arbitrary binary data corresponding to the weights of the model. This data should be close
    // or identical to the data in the original model file, but may be different due to quantization or
    // other optimizations for inference. Any such deviations should be recorded in the metadata or as
    // part of the architecture definition.
    //
    // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
    // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
    // should be padded to `ALIGNMENT` bytes.
    uint8_t tensor_data[];
};
```

## The process of changing the GGUF file during fine-tuning

* Load the original GGUF file.
* Extract the model architecture and initial weights.
* Perform fine-tuning using new data: This involves forward passes, calculating loss, and backpropagation. Gradients are computed and applied to update the weights.
* Update the weights in the GGUF structure.
* Modify relevant metadata.
* Save the updated model back into the GGUF format.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Load the pre-trained model and tokenizer
model_name = "TheBloke/Llama-2-7B-GGUF"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")

# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# Load and preprocess the dataset
dataset = load_dataset("imdb")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
```

## Real-world purpose of fine-tuning

Fine-tuning is done to adapt a pre-trained model to a specific task or domain, improving its performance on that particular application while leveraging its general knowledge.

### Example 1

Let's say we have a large language model trained on general web data, and we want to use it for medical question answering.

#### Before fine-tuning

The model has broad knowledge but might struggle with specific medical terminology or the latest treatment protocols. It could give general answers but might lack precision in medical contexts.

```plaintext
User: "What are the symptoms of acute appendicitis?"
Response: "Acute appendicitis can cause abdominal pain and fever. It's a serious condition that may require medical attention."
```

#### After fine-tuning on medical literature and Q&A datasets

* Improved medical accuracy: The model would provide more precise, up-to-date medical information.

```plaintext
User: "What are the symptoms of acute appendicitis?"
Response: "Common symptoms of acute appendicitis include: 
Sudden pain that begins on the right side of the lower abdomen
Pain that worsens if you cough, walk or make other jarring movements
Nausea and vomiting
Loss of appetite
Low-grade fever that may worsen as the illness progresses
Constipation or diarrhea
Abdominal bloating
If you experience these symptoms, especially severe abdominal pain, seek immediate medical attention."
```
* Better understanding of medical context: The model would better interpret medical jargon and provide more relevant answers.
* Up-to-date information: If fine-tuned on recent medical literature, it could incorporate the latest research and guidelines.
* Improved medical reasoning: The model might better connect symptoms to potential diagnoses or suggest appropriate next steps.
* Domain-specific language: It would use more appropriate medical terminology and phrasing.
* Reduced hallucinations on medical topics: The model would be less likely to generate false or misleading medical information.

* Expected behavior changes:
  * More accurate and detailed responses to medical queries
  * Better handling of medical terminology
  * Improved ability to provide relevant medical advice (with appropriate disclaimers)
  * Potentially faster and more confident responses on medical topics

This example demonstrates how fine-tuning can significantly enhance a model's performance in a specific domain, making it much more useful for specialized applications like medical Q&A. The same principle applies to other domains - legal, financial, technical support, etc. - where adapting a general model to specific knowledge and language can greatly improve its practical utility.

### Example 2

Scenario: You have a pre-trained LLaMA model stored in a GGUF file, and you want to fine-tune it for sentiment analysis on movie reviews. The fine-tuning will involve training the model on a new dataset that includes labeled movie reviews as positive or negative.

#### 1. Original Model Setup

* Model Name: LLaMA-2B
* GGUF File: llama-2b.gguf
* Vocabulary: Contains standard tokens, e.g., ["the", "movie", "was", "good", "bad", ...]
* Model Parameters: Weights and biases for the transformer layers.

#### 2. Prepare Fine-Tuning Data

* Dataset: A collection of 50,000 movie reviews, each labeled as positive or negative.
* New Tokens: The dataset introduces a few new tokens like “cinematography”, which weren’t in the original vocabulary.

#### 3. Fine-Tuning Process

* Training: You fine-tune the model using this dataset. 
* During fine-tuning: 
  * The model’s weights are adjusted based on the new data.
  * The vocabulary is expanded to include the new token “cinematography”.
  * A new output layer might be added for binary classification (positive/negative sentiment).

#### 4. Changes to the Model:

* Updated Weights: The weights in the transformer layers are adjusted.
* Expanded Vocabulary: The vocabulary now includes the new token “cinematography”, and its embedding is initialized and fine-tuned.
* New Classification Layer: A new linear layer is added to map the model’s output to two classes (positive and negative).

#### 5. Update the GGUF File

|Update the GGUF File|Notes|
|-|-|
|**Update Model Parameters**|The updated weights and biases are serialized into the GGUF file. The new classification layer’s weights are also added.|
|**Update Vocabulary and Embeddings**|The new token “cinematography” is added to the vocabulary section. A new embedding vector corresponding to this token is included.|
|**Rebuild GGUF File**|The GGUF file is rebuilt with the new parameters and vocabulary.|
|**The metadata is updated to reflect the fine-tuning process**|Fine-Tuning Dataset: "Movie Reviews Sentiment Analysis Dataset"|
||Date of Fine-Tuning: "2024-08-23"|
||Task: Sentiment Analysis|
||Special Tokens: Added “cinematography”.|
||Recalculate Checksums: A new checksum is calculated to ensure the file’s integrity after the updates.|
|**Save the New GGUF File**|The new file is saved as llama-2b-sentiment.gguf.|

#### 6. Using the Fine-Tuned Model

* Deployment: The fine-tuned GGUF file llama-2b-sentiment.gguf is now ready for deployment.
* Inference: You can load this model and use it to classify new movie reviews as positive or negative.

In this example, you start with a pre-trained LLaMA model in a GGUF file. After fine-tuning the model on a new dataset for sentiment analysis, the GGUF file is updated to reflect the changes in model weights, vocabulary, and structure. The fine-tuned GGUF file is then ready for deployment, enabling the model to perform the new task effectively. This simple example illustrates how fine-tuning affects the model and its GGUF representation.