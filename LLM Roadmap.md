# LLM Roadmap

|Roadmap Section|Subsection|Topics|
|-|-|-|
|LLM Fundamentals Roadmap|Mathematics for Machine Learning|Linear Algebra, Calculus, Probability and Statistics|
||Python for Machine Learning|Python Basics, Data Science Libraries, Data Preprocessing|
||Machine Learning|Supervised Learning, Unsupervised Learning, Model Evaluation|
||Neural Networks|Fundamentals, Training and Optimization, Overfitting, Implement an MLP|
||Natural Language Processing|Text Preprocessing, Feature Extraction Techniques, Word Embeddings, Recurrent Neural Networks|
|LLM Scientist Roadmap|The LLM Architecture|High-level view, Tokenization, Attention mechanisms, Text generation|
||Building an Instruction Dataset|Alpaca-like dataset, Advanced techniques, Filtering data, Prompt templates|
||Pre-training Models|Data pipeline, Causal language modeling, Scaling laws, High-Performance Computing|
||Supervised Fine-Tuning|Full fine-tuning, LoRA, QLoRA, DeepSpeed|
||Preference Alignment|Preference datasets, Proximal Policy Optimization, Direct Preference Optimization|
||Evaluation|Traditional metrics, General benchmarks, Task-specific benchmarks, Human evaluation|
||Quantization|Base techniques, GGUF and llama.cpp, GPTQ and EXL2, AWQ|
||New Trends|Positional embeddings, Model merging, Mixture of Experts, Multimodal models|
|LLM Engineer Roadmap|[Running LLMs](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#running-llms)|LLM APIs, Open-source LLMs, Prompt engineering, Structuring outputs|
||[Building a Vector Store](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#building-a-vector-store)|Ingesting documents, Splitting documents, Embedding models,Vector databases|
||[RAG](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#rag)|Orchestrators, Retrievers, Memory, Evaluation|
||Advanced RAG|Query construction, Agents and tools, Post-processing, Program LLMs|
||Inference Optimization|Flash Attention, Key-value cache, Speculative decoding|
||Deploying LLMs|Local deployment, Demo deployment, Server deployment, Edge deployment|
||Securing LLMs|Prompt hacking, Backdoors, Defensive measures|

---

## LLM Fundamentals Roadmap

---

## LLM Scientist Roadmap

---

## LLM Engineer Roadmap

### Running LLMs
* The Large Language Model API stands as a technical interaction with sophisticated AI systems capable of processing, comprehending, and generating human language. These APIs act as a channel between the intricate algorithms of LLM performance and various applications, enabling seamless integration of language processing functionalities into software solutions.

|Running LLMs||
|-|-|
|LLM APIs|[OpenAI](https://platform.openai.com/), [Google](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview), [Anthropic](https://docs.anthropic.com/claude/reference/getting-started-with-the-api), [Cohere](https://docs.cohere.com/docs)|
||Open-source LLMs API: LLMs: [OpenRouter](https://openrouter.ai/), [Hugging Face](https://huggingface.co/inference-api), [Together AI](https://www.together.ai/)|
|Open Source LLMS|[Hugging Face Hub](https://huggingface.co/models)|
||Run it in Web: [Hugging Face Spaces](https://huggingface.co/spaces)|
||Run it in AP: [LM Studio](https://lmstudio.ai/)|
||Run it in CLI(Command Line Interface) : [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.ai/).|
|Prompt engineering|[Prompt engineering guide](https://www.promptingguide.ai/)|
|Structuring outputs|LLM tasks require a structured output, example: a JSON format. [LMQL](https://lmql.ai/), [Outlines](https://github.com/outlines-dev/outlines), [Guidance](https://github.com/guidance-ai/guidance)

### Building a Vector Store
* Vector stores contain embedding vectors of ingested document chunks. A vector store takes care of storing embedded data and performing vector search for you.
 
|Building a Vector Store|Example: [LocalGPT](https://github.com/PromtEngineer/localGPT)|
|-|-|
|Ingesting documents|`ingest.py`: Ingest or feed information from local files into the knowledge base of local LLM.|
|Splitting documents|Whether in the form of text, PDF, CSV, or Excel files, pass them on to `LangChain` `text_splitter`, so each of the document is going to be divided into multiple chunks.|
|Embedding model|Compute embeddings for each chunk and create a semantic index using `ChromaDB`.|
|Vector databases|([Chroma](https://www.trychroma.com/), [Pinecone](https://www.pinecone.io/), [Milvus](https://milvus.io/), [FAISS](https://faiss.ai/), [Annoy](https://github.com/spotify/annoy), etc.) are designed to store embedding vectors.|

### RAG
* Retrieval augmented generation (RAG),	combines a Large Language Model with external knowledge retrieval.

