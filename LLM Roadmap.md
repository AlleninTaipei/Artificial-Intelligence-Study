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
||[RAG](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#rag)|Framworks, Retrievers, Conversational memory, Evaluation|
||[Advanced RAG](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#advanced-rag)|Query construction, Agents and tools, Pre-retrieval and Post-retrieval, Program LLMs|
||[Inference Optimization](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#inference-optimization)|Flash Attention, Key-value cache, Speculative decoding|
||[Deploying LLMs](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#deployinh-llms)|Local deployment, Demo deployment, Server deployment, Edge deployment|
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
* Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. Retrieval-Augmented Generation (RAG) has emerged as a promising solution by incorporating knowledge from external databases. This enhances the accuracy and credibility of the generation, particularly for knowledge-intensive tasks, and allows for continuous knowledge updates and integration of domain-specific information. RAG synergistically merges LLMs' intrinsic knowledge with the vast, dynamic repositories of external databases.

|RAG||
|-|-|
|Framworks|[LangChain](https://python.langchain.com/docs/get_started/introduction), [LlamaIndex](https://docs.llamaindex.ai/en/stable/)|
|Retrieval|The RAG framework's efficiency hinges on optimizing retrieval sources, granularity, indexing, query handling, and embedding models. Advances in these areas continue to enhance the performance and applicability of LLMs across diverse tasks and domains.|
|Conversational memory|Conversational memory is how a chatbot can respond to multiple queries in a chat-like manner. It enables a coherent conversation, and without it, every query would be treated as an entirely independent input without considering past interactions.|
|Evaluation|[Ragas](https://github.com/explodinggradients/ragas/tree/main) and [DeepEval](https://github.com/confident-ai/deepeval)|


### Advanced RAG

* Naive RAG mainly consists of three parts: indexing, retrieval and generation. Advanced RAG proposes multiple optimization strategies around pre-retrieval and post-retrieval, with a process similar to the Naive RAG, still following a chain-like structure.

|Advanced RAG|Example: [LangChain - OpenAI's RAG](https://blog.langchain.dev/applying-openai-rag/)|
|-|-|
|Query construction|Query construction is taking a natural language query and converting it into the query language of the database you are interacting with.<br>[LangChain - Query Construction](https://blog.langchain.dev/query-construction/)|
|Agents and tools|We can use tools like SQL agents to recover from errors.|
|Pre-retrieval|The primary focus is on optimizing the indexing structure and the original query.|
|Post-retrieval|Once relevant context is retrieved, it’s crucial to integrate it effectively with the query. The main methods in post-retrieval process include rerank chunks and context compressing. Re-ranking the retrieved information to relocate the most relevant content to the edges of the prompt is a key strategy.<br>[RAG-fusion](https://github.com/Raudaschl/rag-fusion)|
|Program LLMs|Framework [DSPy](https://github.com/stanfordnlp/dspy) is a framework for algorithmically optimizing LM prompts and weights, especially when LMs are used one or more times within a pipeline.|

### Inference Optimization

* Basic inference is slow because LLMs have to be called repeatedly to generate the next token. The input sequence increases as generation progresses, which takes longer and longer for the LLM to process. LLMs also have billions of parameters, making it a challenge to store and handle all those weights in memory.

|Inference Optimization||
|-|-|
|Flash Attention|A known issue with transformer models is that the self-attention mechanism grows quadratically in compute and memory with the number of input tokens. This limitation is only magnified in LLMs which handles much longer sequences. To address this FlashAttention and FlashAttention-2 break up the attention computation into smaller chunks and reduces the number of intermediate read/write operations to GPU memory to speed up inference.<br>[Optimizing LLMs for Speed and Memory](https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization)|
|Key-value cache|KV caching, the decode phase generates a single token at each time step, but each token depends on the key and value tensors of all previous tokens.<br>[Multi-Query Attention](https://arxiv.org/abs/1911.02150) (MQA) and [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) (GQA)|
|Speculative decoding|Each input token you need to load the model weights each time during the forward pass. Speculative decoding alleviates this slowdown by using a second smaller and faster assistant model to generate candidate tokens that are verified by the larger LLM in a single forward pass.|

### Deploying LLMs

|Deploying LLMs||
|-|-|
|Local deployment||
|Demo deployment||
|Server deployment||
|Edge deployment||