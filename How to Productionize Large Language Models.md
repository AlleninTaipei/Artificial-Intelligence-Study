# How to Productionize Large Language Models (LLMs)

Understand LLMOps, architectural patterns, how to evaluate, fine tune & deploy HuggingFace generative AI models locally or on cloud.

## Table of contents

|Contents|Items|Notes|
|-|-|-|
|LLMs primer|Transformer architecture|Inputs (token context window)<br>Embedding<br>Encoder<br>Self-attention(multi-head) layers<br>Decoder<br>Softmax output<br>|
||Difference between various LLMs (architecture, weights and parameters)|
||HuggingFace, the house of LLMs||
|How to play with LLMs|Model size and memory needed||
||Local model inference|Quantization<br>Transformers<br>GPT4All<br>LM Studio<br>llama.cpp<br>Ollama
||Google colab||
||AWS|SageMaker Studio, SageMaker Studio Lab, SageMaker Jumpstart, Amazon Bedrock|
||Deploy HuggingFace model on SageMaker endpoint||
|Architectural Patterns for LLMs|Foundation models||
||Prompt engineering|Tokens<br>In-Context Learning<br>Zero-Shot inference<br>One-shot inference<br>Few-shot inference|
||Retrieval Augmented Generation (RAG)|RAG Workflow<br>Chunking<br>Document Loading and Vector Databases<br>Document Retrieval and reranking<br>Reranking with Maximum Marginal Relevance|
||Customize and Fine-tuning|Instruction fine-tuning<br>Parameter efficient fine-tuning<br>LoRA and QLoRA|
||Reinforcement learning from human feedback (RLHF)|Reward model<br>Fine-tune with RLHF|
||Pretraining (creating from scratch)|Continous pre-training<br>Pretraining datasets<br>HuggingFace autotrain|
||Agents|Agent orchestration<br>Available agents|
|Evaluating LLMs|Classical and deep learning model evaluation|Metrics<br>NLP metrics|
||Holistic evaluation of LLMs|Metrics<br>Benchmarks and Datasets<br>Evaluating RLHF fine-tuned model<br>Evaluation datasets for specialized domains|
||Evaluating in CI/CD|Rule based<br>Model graded evaluation|
||Evaluation beyond metrics and benchmarks|Cost and memory<br>Latency<br>Input context length and output sequence max length|
|Deploying LLMs|Deployment vs productionization||
||Classical ML model pipeline|Open-source tools<br>AWS SageMaker Pipelines<br>Different ways to deploy model on SageMaker<br>BYOC (Bring your own container)<br>Deploying multiple models|
||LLM Inference with Quantization|Quantize with AutoGPTQ<br>Quantize with llama.cpp|
||Deploy LLM on Local Machine|llama.cpp<br>Ollama<br>Transformers<br>text-generation webui by oobabooga<br>Jan.ai<br>GPT4ALL<br>Chat with RTX by Nvidia|
|||Deploy LLM on cloud|Major cloud providers<br>Deploy LLMs from HuggingFace on Sagemaker Endpoint<br>Sagemaker Jumpstart<br>SageMaker deployment of LLMs that you have pretrained or fine-tuned|
||Deploy using containers|Benefits of using containers<br>GPU and containers<br>Using Ollama|
||Using specialized hardware for inference|AWS Inferentia<br>Apple Neural engine|
||Deployment on edge devices|Different types of edge devices<br>TensorFlow Lite<br>SageMaker Neo<br>ONNX|
||CI/CD Pipeline for LLM based applications|Fine-tuning Pipeline|
||Capturing endpoint statistics|Ways to capture endpoint statistics<br>Cloud provider endpoints|
|Productionize LLM based projects|An Intelligent QA chatbot powered by Llama 2 Chat<br>LLM based recommendation system chatbot<br>Customer support chatbot using agents||
|Upcoming|GPT-5<br>Prompt compression<br>LLMops<br>AI Software Engineers (or agents)||

### Transformer architecture

Input prompt is stored in a construct called the **input context window**. It is measured by the number of tokens it holds. The size of the context window varies widely from model to model.

**Embeddings** in Tranformers are learned during model pretraining and are actually part of the larger Transformer architeture. Each input token in the context windows is mapped to an embedding. These embeddings are used throughout the rest of the Transformer neural network, including the self-attention layers.
**Embeddings** are used to capture semantic relationships and contextual information.

**Encoder** projects sequence of input tokens into a vector space that represents that strucute and meaning of the input. The vector space representation is learned during model pretraining.

**Self-attention** enables the model to weigh the significance of different words in a sequence relative to each other. This allows the model to capture diverse relationships and dependencies within the sequence, enhancing its ability to understand context and long-range dependencies.
It calculates n square pairwise attention scores between every token in the input with every other token.
Standard attention mechanism uses High Bandwidth Memory (HBM) to store, read and write keys, queries and values. HBM is large in memory, but slow in processing, meanwhile SRAM is smaller in memory, but faster in operations. It loads keys, queries, and values from HBM to GPU on-chip SRAM, performs a single step of the attention mechanism, writes it back to HBM, and repeats this for every single attention step. Instead, Flash Attention loads keys, queries, and values once, fuses the operations of the attention mechanism, and writes them back.

The attention weights are passed throgh rest of the Transformer neural network, including **the decoder. The decoder** uses the attention-based contextual understanding of the input tokens to generate new tokens, which ultimately “completes” the provided input. That is why the base model’s response is often called a completion.

**The softmax output layer** generates a probability distribution across the entire token vocabulary in which each token is assigned a probability that it will be selected text.
Typically the token with highest probability will be generarted as the next token but there are mechanisms like *temperature, top-k & top-p* to modify next token selection to make the model more or less creative.

### Difference between various LLMs

**Architecture**
Encoder only — or autoencoders are pretrained using a technique called masked language modeling (MLM), which randomly mask input tokens and try to predict the masked tokens.
Encoder only models are best suited for language tasks that utilize the embeddings generated by the encoder, such as semantic similarity or text classification because they use bidirectional representations of the input to better understand the fill context of a token — not just the previous tokens in the sequence. But they are not particularly useful for generative tasks that continue to generate more text.
Example of well known encode-only models is BERT.

Decoder only — or autoregressice models are pretrained using unidirectional causal language modeling (CLM), which predicts the next token using only the previous tokens — every other token is masked.
Decoder-only, autoregressive models use millions of text examples to learn a statistical language representation by continously predicting the next token from the previous tokens. These models are the standard for generative tasks, including question-answer. The families of GPT-3, Falcon and Llama models are well-known autoregressive models.

Encoder-decoder — models, often called sequence-to-sequence models, use both the Transformer encoder and decoder. They were originally designed for translation, are also very useful for text-summarization tasks like T5 or FLAN-T5.

**Weights** - In 2022, a group of researchers released a paper that compared model performance of various model and dataset size combinations. The paper claim that the optimal training data size (measured in tokens) is 20x the number of model parameters and that anything below that 20x ration is potentially overparameterized and undertrained.

According to Chinchilla scaling laws, there 175+ billion parameter models (like GPT-3) should be trained on 3.5 trillion tokens. Instead, they were trained with 180–350 billion tokens — an order of magnitude smaller than recommended. 
Llama 2 70 billion parameter model, was trained with 2 trillion tokens — greater than the 20-to-1 token-to-parameter ration described by the paper. This is one of the reason Llama 2 outperformed original Llama model based on various benchmarks.

Attention layers & Parameters (top k, top p) — most of the model cards explain the type of attention layers the model has and how your hardware can exploit it to full potential. Most common open-source models also document the parameters that can be tuned to achieve optimum performance based on your dataset by tuning certain parameters.

### HuggingFace, the house of LLMs

Hugging Face is a platform that provides easy access to state-of-the-art natural language processing (NLP) models, including Large Language Models (LLMs), through open-source libraries. It serves as a hub for the NLP community, offering a repository of pre-trained models and tools that simplify the development and deployment of language-based applications.

### Model sizes and memory needed

A single-model parameter, at full 32-bit precision, is represented by 4 bytes. Therefore, a 1-billion parameter model required 4 GB of GPU RAM just to load the model into GPU RAM at full precision. If you want to train the model, you need more GPU memory to store the states of the numerical optimizer, gradients, and activations, as well as temporary variables used by the function. So to train a 1-billion-parameter model, you need approximately 24GB of GPU RAM at 32-bit full precision, six times the memory compared to just 4GB of GPU RAM for loading the model.

|RAM needed to train a model|Size per paramater|
|-|-|
|Model Parameters|4 bytes|
|Adam optimizer(2 states)|8 bytes|
|Gradients|4 bytes|
|Activations and temp memory vairable size|8 bytes(est.)|
|Total|4 + 20 bytes|

**1 billion parameter model X 4 = 4GB for inference**
**1 billion parameter model X 24 = 24GB for pretrainig in full precision**

### Local model inference

**Quantization** reduces the memory needed to load and train a model by reducing the precision of the model weights. Quantization converts the model parameters from 32-bit precision down to 16-bit precision or even 8-bit, 4-bit or even 1-bit.

**Transformers Library** (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between TensorFlow 2.0 and PyTorch.
Also helps in pre-processing, training, fine-tuning and deploying transformers.

|APIs and tools|common tasks|
|-|-|
|Natural Language Processing|text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation|
|Computer Vision|image classification, object detection, and segmentation|
|Audio|automatic speech recognition and audio classification|
|Multimodal|table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering|

**GPT4All** is a free-to-use, locally running, privacy-aware chatbot which does not require GPU or even internet to work on your machine (or even cloud).
In complete essence, GPT4All is an ecosystem to train and deploy powerful and customized large language models that run locally on consumer grade CPUs. [Nomic AI](https://www.nomic.ai/) supports and maintains this software ecosystem to enforce quality and security alongside spearheading the effort to allow any person or enterprise to easily train and deploy their own on-edge large language models.

**LM Studio** helps you find, download, experiment with LLMs and run any ggml-compatible model from Hugging Face, and provides a simple yet powerful model configuration and inferencing UI.
The app leverages your GPU when possible and you can also choose to offload only some model layers to GPU VRAM.
“GG” refers to the initials of its originator [Georgi Gerganov](https://ggerganov.com/).

**LLaMA.cpp** was a C/C++ port of Facebook’s LLaMA model, a large language model (LLM) that can generate text, translate languages, write different kinds of creative content, and answer your questions in an informative way. 
Now it it not limited to LlaMa family of models. [llama.cpp supports](https://github.com/AlleninTaipei/LLM-Deployment) inference, fine-tuning, pretraining and quantization with minimal setup and state-of-the-art performance on a wide variety of hardware.

**Ollama** is UI based tool that supports inference on number of open-source large language models. It is super easy to install and get running in few minutes.

```plaintext
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Install the latest PowerShell for new features and improvements! https://aka.ms/PSWindows

PS C:\Users\Ryzen> ollama.exe
Usage:
  ollama [flags]
  ollama [command]

Available Commands:
  serve       Start ollama
  create      Create a model from a Modelfile
  show        Show information for a model
  run         Run a model
  pull        Pull a model from a registry
  push        Push a model to a registry
  list        List models
  ps          List running models
  cp          Copy a model
  rm          Remove a model
  help        Help about any command

Flags:
  -h, --help      help for ollama
  -v, --version   Show version information

Use "ollama [command] --help" for more information about a command.
PS C:\Users\Ryzen>
```

**Google Colab** is a cloud-based platform that offers free access to Jupyter notebook environments with GPU and TPU support. You can run Python code, execute commands, and perform data analysis tasks directly in the browser without the need for any setup or installation. For playing with large language models that require GPUs for inference, Colab offers significant advantages with free and paid plans. 

**AWS** [Bedrock, PartyRock, SageMaker: Choosing the right service for your Generative AI applications](https://community.aws/content/2ZG5ag53cbTljStVUn1bbVpWLki/bedrock-partyrock-sagemaker-choosing-the-right-service-for-your-generative-ai-applications)
Amazon EC2 p4d.24xlarge instances have upto 8 GPUs each with Nvidia A100 GPUs and 640 GB of total GPU memory, at the price of $32.77 per hour. Depending on your use case it can still be a lot cost effective than trying to create you own GPU cluster like [Nvidia DGX A100 system](https://www.nvidia.com/en-in/data-center/dgx-platform/).

**Deploy HuggingFace model on SageMaker endpoint**, you can quickly depoy (almost) any HuggingFace Large Language Model using the SageMaker infrastructure.

### Foundational models

The model parameters are learned during the training phase — often called pretraining. They are trained on massive amounts of training data — typically over a period of weeks and months using large, distributed clusters of CPUs and GPUs. After learning billions of parameters (a.k.a weights), these foundation models can represent complex entities such as human language, images, videos and audio clips. In most cases, you will not use foundation models as it is because they are text completion models (atleast for NLP tasks). When these models are fine-tuned using Reinforced Learning from Human Feedback (RHLF) they are more safer and adaptive to general tasks like question-answering, chatbot etc.

Llama 2 is a Foundation model.
Llama 2 Chat has been fine-tuned for chat from base Llama 2 foundational model.

|Data is the differentiator for generative AI applications.|Examples|
|-|-|
|Customize to specific business needs|Healthcare — Understand medical terminology and provide accurate responses related to patient’s health|
|Adapt to domain-specific language|Finance — Teach financial & accounting terms to provide good analysis for earnings reports|
|Enhance performance for specific tasks|Customer Service- Improve ability to understand and respond to customer’s inquires and complaints|
|Improve context-awareness in responses|Legal Services — Better understand case facts and law to provide useful insights for attorneys|

### Prompt engineering

What you do with ChatGPT is prompting and the model responds with completion . “Completion” can also be non-text-based depending on the model like image, video or audio. [Deeplearning.ai Prompt Engineering course](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) is short and free.

Large language models process text using **tokens**, which are common sequences of characters found in a set of text. The models learn to understand the statistical relationships between these tokens, and excel at producing the next token in a sequence of tokens. You can use tool (example: [Open.AI Tokenizer](https://platform.openai.com/tokenizer)) to understand how a piece of text might be tokenized by a language model, and the total count of tokens in that piece of text. You might not know which single word will be parted into 2 or more tokens because it can vary from model to model. As a rule of thumb, it is approximated that for 75 english words ~= 100 tokens, i.e. 1.3 token per word. You can use 1.3 multiplier to estimate the cost of services that use token based pricing.

Provide examples to the model as part of the prompt, and we call this **in-context learning**. If you pass one prompt-completion pair into the context, this is called **one-short inference**; if you pass no example at all, this is called **zero-shot inference**. Zero-shot inference is often used to evaluate a model’s ability to perform a task that it hasn’t been explicity trained on or seen examples for. For zero-shot inference, the model relies on its preexisting knowledge and generalization capabilities to make inference or generate appropriate outputs, even when it encounters tasks or questions it has never seen before. Larger models are surprisingly good at zero-shot inference. If we pass number of prompt-completion pairs in the context, it is called **few-shot inference**. With more examples, or shots, the model more closely follows the pattern of the response of the in-context prompt-completion pairs.

### Retrieval Augmented Generation (RAG)

[Recommender System Chatbot with LLMs](https://github.com/mrmaheshrajput/recsys-llm-chatbot/blob/main/README.md)

|RAG Use cases|Notes|
|-|-|
|Improved content quality|helps in reducing hallucinations and connecting with recent knowledge including enterprise data|
|Contextual chatbots and question answering|enhance chatbot capabilities by integrating with real-time data|
|Personalized search|searching based on user previous search history and persona|
|Real-time data summarization|retrieving and summarizing transactional data from databases, or API calls|


There are two common **RAG workflows** to consider — preparation of data from external knowledge sources, then the integration of that data into consuming applications. Data preparation involves the ingestion of data sources as well as the capturing of key metadata describing the data source. If the information source is a PDF, there will be an additional task to extract text from those documents.

**Chunking** breaks down larger pieces of text into smaller segments. It is required due to context window limits imposed by the LLM. For example, if the model only supports 4,096 input tokens in the context window, you will need to adjust the chunk size to account for this limit. [SentenceTransformers](https://www.sbert.net/#) is the go-to Python module for accessing, using, and training state-of-the-art text and image embedding models. 

**Document Loading and Vector Databases**, a common implementation for document search and retrieval, includes storing the documents in a vector store, where each document is indexed based on an embedding vector produced by an embedding model. [Vector DB Comparison](https://superlinked.com/vector-db-comparison?source=post_page-----060a4cb1a169--------------------------------), a [battle-grade vector database](https://docs.google.com/spreadsheets/d/170HErOyOkLDjQfy3TJ6a3XXXM1rHvw_779Sit-KT7uc/edit?pli=1&gid=0#gid=0) specialises in storage and retrieval of high dimensional vectors in a distributed environment. Each embedding aims to capture the semantic or contextual meaning of the data, and semantically similar concepts end up closed to each other (have a small distance between them) in the vector space. As a result, information retrieval involves finding nearby embeddings that are likely to have similar contextual meaning. Depending on the vector store, you can often put additional metadata such as a reference to the original content the embedding was created from along with each vector embedding. Not just storage, vector databases also support different indexing strategies to enable low-latency retrieval from thousand’s of candidates. Common indexing strategies include, [HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/) and [IVFFlat](https://www.timescale.com/blog/nearest-neighbor-indexes-what-are-ivfflat-indexes-in-pgvector-and-how-do-they-work/). Approximate Nearest Neighbors library like [FAISS](https://github.com/facebookresearch/faiss), Annoy can be kept in memory or persisted on disk. If persisted on disk for more than 1 program to update, then it will lead to loss of records or corruption of index. If your team already uses postgres, then pgvector is a good choice against ANN libraries that will need to be self hosted. Vector databases will evolve and can be used for more than 1 purpose. Distributed vector database like [qrant](https://qdrant.tech/), can do semantic search, recommendations (with native api) and much more.

**Document Retrieval and reranking**, once the text from a document has been embedded and indexed, it can then be used to retrieve relevant information by the application. You may want to rerank the similarity results returned from the vector store to help diversify the results beyond just the similarity scores and improve relevance to the input prompt. A popular reranking algorithm that is build into most vector stores is Maximum Marginal Relevance(MMR). MMR aims to maintain relevance to the input prompt but also reduce redundancy in the retrieved results since the retrieved results can often be very similar. This helps to provide context in the augmented prompt that is relvant as well as diverse.

**Reranking with Maximum Marginal Relevance**, encourages diversity in the result set, which allows the retriever to consider more than just the similarity scores, but also include a diversity factor between 0 and 1, where 0 is maximum diversity and 1 is minimum diversity.

### Customize and Fine-tuning

When we adapt foundation models on our custom datasets and use cases, we call this process fine-tuning. There are two main fine-tuning techniques.

In contrast to the billions of tokens needed to pretrain a foundation model, you can achieve very good results with **instruction fine-tuning** using a relatively small instruction dataset — often just 500–1,000 examples is enough. Typically, however, the more examples you provide to the model during fine-tuning, the better the model becomes. To preserve the model’s general-purpose capability and prevent “catastrophic forgetting” in which the model becomes so good at a single task that it may lose its ability to generalize, you should provide the model with many different types of instructions during fine-tuning.

**Parameter-efficient fine-tuning** (PEFT) provides a set of techniques allowing you to fine-tune LLMs while utilizing less compute resources. [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.15647), focuses on freezing all or most of the model’s original parameters and extending or replacing model layers by training an additional, much smaller, set of parameters. The most commonly used techniques fall into the additive and reparameterization categories. Full fine-tuning often requires a large amount of GPU RAM, which quickly increases the overall computing budget and cost. PEFT in some cases, the number of newly trained parameters is just 1–2% of the original LLM weights. Because you’re training a relatively small number of parameters, the memory requirements for fine-tuning become more managable and can be often performed on a single GPU. In addition, PEFT methods are also less prone to catastrophic forgetting, due to the weights of the original foundation model remain frozen, preserving the model’s original knowledge or parametric memory.

**LoRA and QLoRA** reduce the number of trainable parameters and, as a result, the training time required and results in a reduction in the compute and storage resources required. LoRA is also used for multimodel models like Stable Diffusion, which uses a Transformer-based language model to help align text to images. The size of the low-rank matrices is set by the parameters called rank (r). Rank refers to the maximum number of linearly independent columns (or rows) in the weight matrix. A smaller value leads to a simpler low-rank matrix with fewer parameters to train. Setting the rank between 4 and 16 can often provide you with a good trade-off between reducing the number of trainable parameters while preserving acceptable levels of model performance. [QLoRA](https://arxiv.org/pdf/2305.14314) aims to further reduce the memory requirements by combining low-rank adaptation with quantization. QLoRA uses 4-bit quantization in a format called NormalFloat4 or nf4. [Code LoRA from Scratch](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch?source=post_page-----060a4cb1a169--------------------------------)

### Reinforcement learning from human feedback (RLHF)

Reinforcement learning from human feedback (RLHF) is a fine-tuning mechanism that uses human annotation — also called human feedback — to help the model adapt to human values and preferences. For example, you could fine-tune a chat assistant specific to each user of your application. This chat assistant can adopt the style, voice, or sense of humour of each user based on their interactions with your application.

**Reward model** is typically a classifier the predicts one of two classes — positive or negative. Positive refers to the generated text that is more human-aligned, or preferred. Negative class refers to non-preferred response.
To determine what is helpful, honest and harmless(HHH)(positive), you often need a annotated dataset using human-in-the-loop workflow. The reward models are often small binary classifiers and based on smaller language models like [BERT uncased](https://huggingface.co/google-bert/bert-base-uncased), [Distillbert](https://huggingface.co/lvwerra/distilbert-imdb), or [Facebook’s hate speech detector](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target). You can also train your own reward model, however it is a relatively labour-intensive and costly endeavour.

**Fine-tune with RLHF**, there is a popular RL algorithm called Proximal Policy Optimization (PPO) used to perform the actual model weight updated based on the reward value assigned to a given prompt and completion. With each iteration, PPO makes small and bounded updates to the LLM weights — hence the term Proximal Policy Optimization. By keeping the changes small with each iteration, the fine-tuning process is more stable and the resulting model is able to generalize well on new inputs. PPO updates the model weights through backpropagation. After many iterations, you should have a more human-aligned generative model. [RLHF in 2024 with DPO & Hugging Face](https://www.philschmid.de/dpo-align-llms-in-2024-with-trl?source=post_page-----060a4cb1a169--------------------------------)


TBD: Pretraining (creating from scratch)

Continous pre-training<br>Pretraining datasets<br>HuggingFace autotrain|

## Source

[How to Productionize Large Language Models (LLMs) - Mahesh Mar 27, 2024](https://mrmaheshrajput.medium.com/how-to-productionize-large-language-models-llms-060a4cb1a169)
[Hugging Face - Flash Attention](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention)
