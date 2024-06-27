# LLM Roadmap

|Roadmap Section|Subsection|Topics|
|-|-|-|
|LLM Engineer Roadmap|[Running LLMs](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#running-llms)|LLM APIs, Open-source LLMs, Prompt engineering, Structuring outputs|
||[Vector Store](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#vector-store)|Ingesting documents, Splitting documents, Embedding models,Vector databases|
||[RAG](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#rag)|Framworks, Retrievers, Conversational memory, Evaluation|
||[Advanced RAG](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#advanced-rag)|Query construction, Agents and tools, Pre-retrieval and Post-retrieval, Program LLMs|
||[Inference Optimization](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#inference-optimization)|Flash Attention, Key-value cache, Speculative decoding|
||[Deploying LLMs](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#deploying-llms)|Local deployment, Demo deployment, Server deployment, Edge deployment|
||[Securing LLMs](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#securing-llms)|Prompt hacking, Backdoors, Defenses and Evaluations|
|LLM Scientist Roadmap|[The LLM Architecture](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#the-llm-architecture)|Transformer, Tokenization, Attention mechanisms, Text generation|
||[Instruction Dataset](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#instruction-dataset)|Alpaca-like dataset, Advanced techniques, Filtering data, Prompt templates|
||[Pre-training Models](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#pre-traning-models)|Data pipeline, Causal language modeling, Scaling laws, High-Performance Computing|
||[Supervised Fine-Tuning](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#supervised-fine-tuning)|Full fine-tuning, LoRA, QLoRA|
||[Preference Alignment](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#preference-alignment)|Preference datasets, Proximal Policy Optimization, Direct Preference Optimization|
||[Evaluation](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#evaluation)|Traditional metrics, General benchmarks, Task-specific benchmarks, Human evaluation|
||[Quantization](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#quantization)|Base techniques, GGUF and llama.cpp, GPTQ and EXL2, AWQ|
||[New Trends](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#new-trends)|Positional embeddings, Model merging, Mixture of Experts, Multimodal models|
|LLM Fundamentals Roadmap|[Mathematics for Machine Learning](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#mathematics-for-machine-learning)|Linear Algebra, Calculus, Probability and Statistics|
||[Python for Machine Learning](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#python-for-machine-learning)|Python Basics, Data Science Libraries, Data Preprocessing|
||[Machine Learning](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#machine-learning)|Supervised Learning, Unsupervised Learning, Model Evaluation|
||[Neural Networks](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#neural-networks)|Fundamentals, Training and Optimization, Overfitting, Implement an MLP|
||[Natural Language Processing](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLM%20Roadmap.md#natural-language-processing)|Text Preprocessing, Feature Extraction Techniques, Word Embeddings, Recurrent Neural Networks|

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

### Vector Store
* Vector stores contain embedding vectors of ingested document chunks. A vector store takes care of storing embedded data and performing vector search for you.
 
|Vector Store|Example: [LocalGPT](https://github.com/PromtEngineer/localGPT)|
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
|[FlashAttention](https://arxiv.org/abs/2205.14135)|A known issue with transformer models is that the self-attention mechanism grows quadratically in compute and memory with the number of input tokens. This limitation is only magnified in LLMs which handles much longer sequences. To address this FlashAttention and [FlashAttention-2](https://arxiv.org/abs/2307.08691) break up the attention computation into smaller chunks and reduces the number of intermediate read/write operations to GPU memory to speed up inference.<br>[Optimizing LLMs for Speed and Memory](https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization)|
|Key-value cache|KV caching, the decode phase generates a single token at each time step, but each token depends on the key and value tensors of all previous tokens.<br>[Multi-Query Attention](https://arxiv.org/abs/1911.02150) (MQA) and [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) (GQA)|
|Speculative decoding|Each input token you need to load the model weights each time during the forward pass. Speculative decoding alleviates this slowdown by using a second smaller and faster assistant model to generate candidate tokens that are verified by the larger LLM in a single forward pass.|

|FlashAttention|FlashAttention and FlashAttention-2 indeed optimize the attention computation in transformers by breaking it into smaller chunks. This approach reduces the number of intermediate read/write operations to GPU memory, which significantly speeds up both training and inference.|
|-|-|
|FlashAttention|FlashAttention optimizes the standard attention mechanism by leveraging efficient memory usage and computational patterns. The traditional attention mechanism requires computing and storing large intermediate matrices, which leads to significant memory overhead and numerous memory accesses.|
||Chunked Computation:FlashAttention divides the attention computation into smaller, more manageable chunks. By processing these chunks sequentially, it reduces the peak memory usage and keeps intermediate data in fast, on-chip memory (like GPU registers or shared memory), minimizing expensive reads and writes to slower global memory.|
||Memory Coalescing: By structuring memory access patterns to be more coalesced (i.e., accessing contiguous memory locations together), FlashAttention reduces the latency and bandwidth required for memory operations. This is crucial for optimizing performance on GPU architectures.|
||Attention Matrix Sparsity: Exploiting the sparsity in attention matrices (when applicable), FlashAttention can skip computations for negligible attention scores, further saving on computational and memory resources.
|FlashAttention-2|FlashAttention-2 builds upon the optimizations introduced in FlashAttention and incorporates additional improvements to further enhance performance:|
||Better Load Balancing: FlashAttention-2 introduces more sophisticated load balancing techniques, ensuring that all computational resources are utilized more effectively. This prevents some processing units from being idle while others are overloaded.|
||Improved Chunking Strategy:The chunking strategy in FlashAttention-2 is refined to better align with the hardware’s memory hierarchy. This leads to even more efficient use of on-chip memory and reduces the frequency and volume of memory transfers to and from global memory.|
||Enhanced Sparsity Handling:FlashAttention-2 improves the handling of sparse attention matrices, with more advanced algorithms to identify and exploit sparsity patterns, reducing unnecessary computations.|
||Pipelining and Parallelism:FlashAttention-2 increases the degree of parallelism and introduces pipelining techniques, allowing multiple stages of the attention computation to overlap. This reduces the overall latency and increases throughput.|

|Impact on Performance|FlashAttention and FlashAttention-2 significantly reduce the number of intermediate memory operations, which are a major bottleneck in standard transformer models.|
|-|-|
|Training Phase|During training, the model repeatedly computes gradients and updates weights based on a large amount of data. This involves significant computational overhead and memory usage, especially for large models like transformers.|
||**Reduced Memory Usage:** By chunking the attention computation and keeping intermediate results in fast on-chip memory, these techniques significantly reduce the overall memory footprint. This allows for training larger models or using larger batch sizes within the same memory constraints.|
||**Faster Computation:** The reduced number of memory accesses and efficient use of GPU resources speed up the attention mechanism, which is one of the most computationally intensive parts of the transformer. This results in faster training times.|
||**Gradient Computation:** During backpropagation, gradients need to be computed for the attention weights and inputs. FlashAttention’s efficient handling of these computations reduces the overhead associated with gradient calculations, leading to more efficient training.|
|Inference Phase|During inference, the model processes new data to generate predictions. The efficiency of this phase is crucial for real-time applications and large-scale deployments.|
||**Low Latency:** The reduced number of memory operations and the efficient chunking strategy lower the latency of the attention computation, making inference faster.|
||**Scalability:** Improved memory usage and computational efficiency enable the deployment of larger models in resource-constrained environments, such as on edge devices or in situations where memory and computational power are limited.|
||**Energy Efficiency:** By optimizing memory access patterns and reducing unnecessary computations, these techniques also help in reducing the energy consumption of the models during inference, which is important for sustainable AI applications.|
|Summary|FlashAttention and FlashAttention-2 improve the efficiency of both training and inference by optimizing the attention computation. This includes reducing memory usage, speeding up computations, and making better use of GPU resources. These benefits are crucial for handling large models and datasets, improving training times, and enabling faster and more efficient inference.|

|FlashAttention and FlashAttention-2 Optimizations|FlashAttention and FlashAttention-2 do not specifically leverage NVMe SSDs for their optimizations. Instead, their primary focus is on improving memory and computational efficiency within the GPU's architecture.|
|-|-|
|On-chip Memory Utilization|These techniques focus on maximizing the use of fast on-chip memory, such as GPU registers and shared memory. By keeping intermediate computations within the GPU’s fast memory, they reduce the need to access slower global memory, thereby speeding up the computation.|
|Chunking Strategy|The attention mechanism is broken down into smaller chunks to fit within the on-chip memory. This minimizes data transfers between the GPU and its global memory, which is one of the main performance bottlenecks.|
|Efficient Memory Access Patterns|By optimizing how data is accessed and stored during computation, FlashAttention and FlashAttention-2 reduce memory latency and bandwidth usage. This includes techniques like memory coalescing and exploiting sparsity in attention matrices.|

|Role of NVMe SSDs|NVMe SSDs are high-speed storage devices that offer significantly faster read and write speeds compared to traditional HDDs and even older SSDs.|
|-|-|
|Reduce Data Loading Times|NVMe SSDs can dramatically speed up the loading of large datasets and models from storage to RAM. This is particularly beneficial during the initial stages of training or inference when large amounts of data need to be loaded into memory.|
|Improve I/O Performance|For applications that require frequent access to large datasets stored on disk, NVMe SSDs can reduce the time spent on I/O operations.|
|**Why FlashAttention Doesn’t Directly Use NVMe SSDs**||
|GPU-Centric Optimization|FlashAttention and FlashAttention-2 are designed to optimize the computational efficiency of the GPU during the attention mechanism computation. The main bottlenecks they address are related to the memory hierarchy and computational patterns within the GPU, not the storage subsystem.|
|Memory Hierarchy Focus|These techniques improve performance by optimizing how data is handled once it is already in the GPU's memory, rather than optimizing the process of getting data from storage to the GPU. The GPU memory (both on-chip and global) is much faster than even the fastest NVMe SSDs, so the primary focus is on making the best use of this high-speed memory.|
|Conclusion|FlashAttention and FlashAttention-2 optimize the internal GPU memory and computation processes rather than leveraging NVMe SSDs. While NVMe SSDs are beneficial for quickly loading data and models into RAM, FlashAttention’s optimizations are targeted at reducing memory and computation bottlenecks within the GPU itself. Thus, their performance improvements come from more efficient use of the GPU's internal resources, not from enhancements in data storage or retrieval from NVMe SSDs.|

|Techniques Utilizing NVMe SSDs in LLM Tasks|NVMe SSDs can be utilized in large language model (LLM) tasks to enhance performance, particularly when dealing with large datasets and models.|
|-|-|
|Memory Mapping Large Models|Memory mapping allows a large model to be loaded into virtual memory, where portions of the model are loaded into physical RAM on demand. NVMe SSDs, with their high read speeds, can significantly reduce the time taken to access parts of the model that are not currently in RAM. This is particularly useful for very large models that exceed the available system RAM.|
|Data Preprocessing and Caching|Preprocessing large datasets often involves multiple I/O operations. Using NVMe SSDs can speed up this process, reducing the time taken to read and write large volumes of data. Additionally, intermediate results can be cached on NVMe SSDs to avoid redundant computations, leveraging their fast read/write capabilities.|
|Distributed Training|In distributed training setups, NVMe SSDs can be used to quickly load data onto different nodes in a cluster. This minimizes the data transfer bottleneck and allows faster synchronization between nodes.|
|Checkpointing|Frequent checkpointing during model training ensures that progress is saved and can be resumed in case of interruptions. NVMe SSDs reduce the overhead associated with writing large checkpoint files, making the process faster and less disruptive to the training workflow.|
|Hybrid Memory Systems|Some systems use a combination of RAM and NVMe SSDs to extend the effective memory available for large models. This setup allows less frequently accessed data to reside on the NVMe SSD, while keeping the most critical data in faster RAM. Techniques such as paging or tiered storage can be employed to manage this hybrid memory.|
|**Practical Applications and Benefits**||
|Inference Speed-Up|For inference tasks, particularly in real-time applications, using NVMe SSDs to store models and data can reduce latency compared to traditional storage solutions. This is crucial for applications requiring quick responses, such as conversational AI and search engines.|
|Handling Large Datasets|When dealing with massive datasets that do not fit entirely in RAM, NVMe SSDs can act as a high-speed buffer. This is especially useful for data-intensive tasks like training LLMs on extensive text corpora, where frequent data access is necessary.|
|Model Loading and Initialization|Loading large models from storage can be time-consuming. NVMe SSDs speed up this process, reducing the initialization time for inference or training sessions. This is beneficial when deploying models in environments where quick startup times are essential.|
|Data Augmentation and Pipeline Efficiency|Data augmentation techniques, which often require loading and processing large amounts of data, benefit from the high throughput of NVMe SSDs. This results in faster data pipelines and more efficient use of computational resources.|
|**Example Implementations**||
|TorchElastic|In PyTorch's distributed training framework, TorchElastic, NVMe SSDs can be used to store and quickly access datasets and checkpoints, improving the efficiency of distributed training jobs.|
|Hugging Face Transformers|When fine-tuning or deploying transformers from the Hugging Face library, models and datasets can be stored on NVMe SSDs to accelerate loading times and reduce latency during inference.|
|Conclusion|While FlashAttention and FlashAttention-2 optimize GPU memory and computation, NVMe SSDs provide significant benefits for LLM tasks related to data loading, preprocessing, and checkpointing. Leveraging NVMe SSDs can lead to faster initialization, reduced latency, and more efficient data handling, making them a valuable component in the overall infrastructure for training and deploying large language models.|

|Flash-Decoding|A technique designed to accelerate the decoding process in transformer-based language models, particularly during inference. This method focuses on optimizing memory access and computational efficiency to reduce the latency and computational overhead of generating text.|
|-|-|
|Chunked Decoding|Similar to FlashAttention, Flash-Decoding breaks the decoding process into smaller chunks. This helps in managing memory more efficiently by processing smaller pieces of data at a time, reducing the overall memory footprint.|
|Efficient Memory Utilization|By keeping intermediate computations within the GPU's fast on-chip memory (such as registers and shared memory), Flash-Decoding minimizes the need for slow memory transfers between the GPU and global memory.|
|Parallel Processing|Flash-Decoding leverages the parallel processing capabilities of modern GPUs. By organizing computations in a way that maximizes parallelism, it ensures that all available GPU cores are utilized effectively, speeding up the decoding process.|
|Reduced Latency|The technique is specifically designed to lower the latency of generating each token during inference. This is crucial for real-time applications where quick response times are essential, such as conversational AI and real-time translation systems.|

|How Flash-Decoding Works|[Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)|
|-|-|
|Input Token Processing|During the decoding phase, the model processes input tokens to generate the next token in the sequence. Flash-Decoding optimizes this step by using chunking and parallel processing to handle multiple tokens simultaneously.|
|Intermediate Computations|Intermediate results, such as the logits for each token, are computed and stored in fast on-chip memory. This reduces the number of read/write operations to slower global memory, which can be a significant bottleneck.|
|Output Token Generation|The next token is generated based on the processed input tokens and intermediate computations. This process is repeated iteratively to generate the entire sequence, with each step being optimized for speed and memory efficiency.|
|**Practical Implications**||
|Faster Inference|By reducing the computational overhead and latency associated with the decoding process, Flash-Decoding enables faster inference times. This is particularly beneficial for applications that require rapid generation of text, such as chatbots, virtual assistants, and interactive storytelling.|
|Resource Efficiency|Optimizing memory usage and computational efficiency means that the same hardware can handle larger models or more concurrent requests, improving the overall throughput of the system.|
|Scalability|Flash-Decoding can be scaled to work with larger models and datasets, making it suitable for state-of-the-art language models that require significant computational resources.|
|**Comparison with FlashAttention**|While FlashAttention focuses on optimizing the attention mechanism within transformers, Flash-Decoding targets the decoding phase. Both techniques share similar principles, such as chunking and efficient memory utilization, but they are applied to different parts of the model's operation:|
|FlashAttention|Optimizes the computation of attention weights and the resulting attention output.|
|Flash-Decoding|Optimizes the generation of tokens during the decoding phase, reducing latency and improving throughput.|
|Conclusion|Flash-Decoding is a powerful technique for accelerating the inference phase of transformer-based language models. By optimizing memory access patterns and leveraging the parallel processing capabilities of GPUs, it reduces latency and improves computational efficiency, making it an essential tool for real-time applications and large-scale deployments of language models.|

* When comparing different techniques used in Large Language Models (LLMs) for their VRAM size efficiency and training time performance, it's important to consider a variety of methods and optimizations.

|VRAM Size Efficiency|Efficiency Ranking|Reason|
|-|-|-|
|FlashAttention/FlashAttention-2|High|By breaking the attention computation into smaller chunks and reducing intermediate read/write operations, FlashAttention optimizes memory usage, significantly lowering the VRAM requirements.|
|[Mixed Precision Training](https://arxiv.org/abs/1710.03740) (FP16/BFloat16)|High|Mixed precision training uses 16-bit floats instead of 32-bit floats, effectively halving the memory requirements for storing weights and activations, thus reducing VRAM usage.|
|Gradient Checkpointing|High|This technique saves memory by recomputing intermediate activations during the backward pass instead of storing them, reducing VRAM consumption at the cost of additional computation.|
|Memory Mapping and Swapping (Offloading)|Moderate|Techniques like ZeRO-Offload (used in DeepSpeed) offload certain parts of the model to CPU memory, reducing VRAM usage but potentially increasing data transfer overhead.|
|Model Parallelism (Tensor and Pipeline Parallelism)|Variable|Distributes the model across multiple GPUs to fit larger models into memory. The efficiency depends on the inter-GPU communication overhead and the model architecture.|

|Training Time Performance|Result|Reason|
|-|-|-|
|Mixed Precision Training (FP16/BFloat16)|Improved|Reduces the amount of data being processed per operation, leading to faster computations and reduced training times, especially on hardware that supports native mixed-precision operations.|
|FlashAttention/FlashAttention-2|Improved|Optimizes the attention mechanism to reduce computational overhead and memory bottlenecks, resulting in faster training iterations.|
|Gradient Checkpointing|Increased|While it reduces memory usage, it introduces additional computation during the backward pass to recompute intermediate activations, which can slow down training.|
|Data Parallelism|Improved|Distributes the training data across multiple GPUs, enabling parallel processing and faster training. However, efficiency gains are dependent on the communication overhead between GPUs.|
|Model Parallelism (Tensor and Pipeline Parallelism)|Variable|Can improve training times by utilizing multiple GPUs to handle larger models, but the inter-GPU communication can become a bottleneck, especially for fine-grained parallelism.|

|Technique|VRAM Size Efficiency|Training Time Performance|
|-|-|-|
|FlashAttention/FlashAttention-2|High|Improved|
|Mixed Precision Training (FP16)|High|Improved|
|Gradient Checkpointing|High|Increased|
|Memory Mapping and Swapping|Moderate|Variable|
|Model Parallelism|Variable|Variable|
|Data Parallelism|Moderate|Improved|

* High VRAM Efficiency: Techniques like FlashAttention, mixed precision training, and gradient checkpointing are highly effective at reducing VRAM usage, making them suitable for running larger models on hardware with limited memory.
Improved Training Time: Mixed precision training and FlashAttention stand out for improving training time performance by optimizing computations and memory usage.
Selecting the right technique depends on the specific constraints and requirements of the training environment, such as available hardware resources and the need to balance between memory efficiency and computational speed.

### Deploying LLMs

|Deploying LLMs||
|-|-|
|Local deployment|[LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.ai/), [oobabooga](https://github.com/oobabooga/text-generation-webui), [kobold.cpp](https://github.com/LostRuins/koboldcpp)|
|Demo deployment|[Gradio](https://www.gradio.app/), [Streamlit](https://docs.streamlit.io/)|
|Server deployment|[SkyPilot](https://skypilot.readthedocs.io/en/latest/), [TGI](https://github.com/huggingface/text-generation-inference), [vLLM](https://github.com/vllm-project/vllm/tree/main)|
|Edge deployment|[MLC LLM](https://github.com/mlc-ai/mlc-llm), [mnn-llm](https://github.com/wangzhaode/mnn-llm/blob/master/README_en.md)|

### Securing LLMs

|Securing LLMs||
|-|-|
|Prompt hacking|The central focus of prompt hacking is the strategic manipulation of inputs or prompts given to Language Models (LLMs). These prompts act as instructions or queries to elicit specific responses from the AI system. By carefully crafting prompts that exploit the model's weaknesses, prompt hackers can influence the generated content in ways that may not align with the intended purpose of the AI system. This manipulation can result in misinformation, biased outputs, or even security breaches, depending on the context and intent of the attack.|
|Backdoors|A backdoored language model might be used to spread misinformation, exploit personal data, or influence user decisions based on skewed or biased responses. This kind of vulnerability in language models is particularly concerning due to their widespread use and the growing trust users place in their responses.|
|Defenses and Evaluations|Red-team attacks are effective against unaligned LLMs but are ineffective against LLMs with builtin security.<br>[garak](https://github.com/leondz/garak/), [langfuse](https://github.com/langfuse/langfuse)|

---

## LLM Scientist Roadmap

### The LLM Architecture

#### Transformer

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* Introduced in 2017, replaced the traditional recurrent neural network (RNN) architecture in machine translation tasks and became the state-of-the-art model. The core components are the Encoder, the Decoder, and the attention mechanism within these modules.
* The Transformer architecture consists of an encoder and a decoder, each of which is composed of several layers. Each layer consists of two sub-layers: a multi-head self-attention mechanism and a feed-forward neural network. The multi-head self-attention mechanism allows the model to attend to different parts of the input sequence, while the feed-forward network applies a point-wise fully connected layer to each position separately and identically.
* The power of the transformer architecture lies in its ability to learn the relevance and context of all of the words in a sentence. Each word next to its neighbor and to every other word in a sentence. To apply attention weights to those relationships so that the model learns the relevance of each word to each other. These attention weights are learned during LLMs training.

#### Tokenization

* [OpenAI Tokenizer](https://platform.openai.com/tokenizer)

#### Attention mechanism

* The attention mechanism was born to help memorize long source sentences in neural machine translation (NMT). Rather than building a single context vector out of the encoder’s last hidden state, the secret sauce invented by attention is to create shortcuts between the context vector and the entire source input. The weights of these shortcut connections are customizable for each output element.
* While the context vector has access to the entire input sequence, we don’t need to worry about forgetting. The alignment between the source and target is learned and controlled by the context vector. Essentially the context vector consumes three pieces of information: encoder hidden states, decoder hidden states and alignment between source and target.
* Self-attention helps the model focus on important parts of the input data by weighing the relevance of different words in a sequence. Involves calculating query, key, and value vectors for words, and using these to determine attention weights through a softmax function. Multi-head attention extends this mechanism by performing multiple self-attention operations in parallel, capturing various aspects of input.

#### Text Generation

* This task covers guides on both text-generation and text-to-text generation models. Popular large language models that are used for chats or following instructions are also covered in this task. You can find the list of selected open-source large language models here, ranked by their performance scores.
* Instruction Models: A model trained for text generation can be later adapted to follow instructions.

### Instruction Dataset

#### Alpaca-like dataset

* [Alpaca 7B](https://crfm.stanford.edu/2023/03/13/alpaca.html)

|Instruction dataset|A list of pairs: instruction and answer|
|-|-|
|Usage|**Designed for fine-tuning LLMS**|
|Create|Often created through self-instruct methods or human curation|
||**Use an existing dataset and convert it into an instruction dataset.**|
||**Use existing LLMs to create an instruction dataset.**|
||**Manually create an instruction dataset.**|

|Dataset|Advanced techniques|
|-|-|
|Self-instruct method|Using a large language model to generate training data for itself.|
||Iterative process of generating instructions, responses, and refining them|
|Data filtering and cleaning|Removing low-quality or inappropriate content|
||Ensuring diversity in tasks and domains|
||Balancing different types of instructions|
|Prompt engineering|Crafting effective instructions and prompts|
||Ensuring clarity and specificity in the tasks|
|Fine-tuning|Adapting pre-trained models on the Alpaca-like dataset|
||Often using techniques like LoRA (Low-Rank Adaptation) for efficiency|
|Evaluation|Using benchmarks to assess the model's instruction-following abilities|
||Human evaluation of generated responses|
|Iterative improvement|Analyzing model outputs to identify weaknesses|
||Expanding the dataset to cover more tasks or improve performance in specific areas|
|Ethical considerations|Ensuring the dataset and resulting model behave ethically|
||Implementing safeguards against generating harmful content|

#### Filtering data

* This involves selectively processing or extracting specific information from a larger dataset based on certain criteria. It's a crucial step in data analysis and manipulation.

|Filtering data||
|-|-|
|Definition|Data filtering is the process of selecting a subset of data from a larger dataset based on specific criteria or conditions.|
|Techniques|Boolean filtering: Using logical operators (AND, OR, NOT) to combine conditions|
||Numeric filtering: Selecting data based on numerical thresholds or ranges|
||Text-based filtering: Using string matching or regular expressions|
||Date/time filtering: Selecting data within specific time periods|
|Applications|Database queries|
||Spreadsheet analysis|
||Data cleaning and preprocessing|
||Business intelligence and reporting|
|Tools|SQL for relational databases|
||Pandas library in Python(e.g.,`pandas.DataFrame.query`, `pandas.DataFrame.loc`)|
||Excel filters and formulas|
||Various data analysis and visualization tools (e.g., Tableau, Power BI)|

#### Prompt templates:

* These are pre-designed structures for generating prompts in AI interactions. They help create consistent and effective prompts for various tasks or queries.

|Prompt templates||
|-|-|
|Definition|Predefined structures or patterns for creating prompts to interact with AI models or generate specific types of content.|
|Components|Fixed text: The unchanging parts of the prompt|
||Variables: Placeholders for dynamic content|
||Instructions: Guidance for the AI model on how to interpret or respond|
|Benefits|Consistency in AI interactions|
||Improved efficiency in prompt creation|
||Better control over AI outputs|
||Easier fine-tuning and optimization of prompts|
|Use cases|Chatbot interactions|
||Content generation (e.g., articles, product descriptions)|
||Code generation and documentation|
||Language translation|
||Data analysis and summarization|
|Implementation|String formatting in various programming languages|
||Dedicated prompt engineering tools and platforms|
||Integration with AI model APIs|

>Example:
>Summarize the following {text_type} in {number} sentences:
>{input_text}

### Pre-training Models

|LLMS|Pre-training Models|
|-|-|
|Data pipeline|A data pipeline refers to the series of processes and tools used to collect, process, and transform raw data into a format suitable for training machine learning models. For language models, this typically involves gathering text data from various sources, cleaning and preprocessing it, and converting it into a format the model can ingest.|
|Causal language modeling|This is a type of language modeling where the task is to predict the next word or token in a sequence, given the previous words. It's "causal" because it only looks at past context to make predictions, not future context. This approach is fundamental to many large language models, including GPT (Generative Pre-trained Transformer) models.|
|[Scaling laws](https://github.com/AlleninTaipei/Artificial-Intelligence-Introduction-for-Beginners/blob/main/LLM%20Parameters%20and%20Memory%20Estimation.md#scaling-laws)|These are empirical observations about how the performance of machine learning models, particularly language models, improves as you increase various factors such as model size, dataset size, and compute resources. Researchers have found that many aspects of model performance follow predictable patterns as these factors are scaled up.|
|High-Performance Computing (HPC)|This refers to the use of powerful computer systems and parallel processing techniques to solve complex computational problems. In the context of large language models, HPC is crucial for training and running these models efficiently, as they require enormous amounts of computational power.|

### Supervised Fine-Tuning

* **Full fine-tuning** involves retraining an entire pre-trained model on a new dataset to adapt it to a specific task or domain. **All model parameters are updated during this process**.
* SFT is a particular way of fine-tuning. It's a subset or specific application of the broader concept of fine-tuning, with an emphasis on supervised learning using task-specific data.
  **Uses labeled data to guide the model's learning**
  **Often involves instruction-following or task-specific training**

|Fine-Tuning||
|-|-|
|[LoRA](https://arxiv.org/abs/2106.09685)|A parameter-efficient fine-tuning method that adds trainable rank decomposition matrices to existing weights. This reduces memory usage and training time compared to full fine-tuning.|
|[QLoRA](https://arxiv.org/abs/2305.14314)|An extension of LoRA that uses quantization techniques to further reduce memory requirements, allowing fine-tuning of large models on consumer-grade hardware.|

* [DeepSpeed](https://www.deepspeed.ai/) is a deep learning optimization library developed by Microsoft. It provides various features to improve training efficiency, including model parallelism, optimizer state partitioning, and gradient accumulation **for multi-GPU and multi-node settings.**|
* [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)  is a tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures.
* [Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments](https://lightning.ai/pages/community/lora-insights/)
* [The Novice's LLM Training Guide](https://rentry.org/llm-training)
* [Fine-Tune Your Own Llama 2 Model in a Colab Notebook](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html)
* [A Beginner’s Guide to LLM Fine-Tuning](https://mlabonne.github.io/blog/posts/A_Beginners_Guide_to_LLM_Finetuning.html)

### Preference Alignment

|Peference Alignment||
|-|-|
|Preference datasets|These are collections of data representing human preferences, often used in machine learning to train models to align with human values or choices. They typically consist of paired comparisons where humans indicate their preference between two options.|
|[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)|PPO is a popular reinforcement learning algorithm developed by OpenAI. It's used to train agents to make decisions in complex environments. PPO aims to improve the stability and efficiency of policy gradient methods by limiting the size of policy updates.|
|[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)|DPO is a more recent approach to training language models using human preference data. It directly optimizes a model to match the preference distribution implied by a dataset of human preferences, without using reinforcement learning techniques.|

* [Synthesize data for AI and add feedback on the fly!](https://github.com/argilla-io/distilabel)
* [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)
* [Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning)

### Evaluation

|Evaluation||
|-|-|
|Traditional metrics|These typically include quantitative measures like accuracy, precision, recall, F1 score, perplexity, and others depending on the specific task.|
|General benchmarks|These are standardized tests designed to assess AI capabilities across a range of tasks. Examples include GLUE, SuperGLUE, and MMLU.|
|Task-specific benchmarks|These focus on evaluating performance on particular applications or domains, such as machine translation, text summarization, or question answering.|
|Human evaluation|This involves having human raters assess model outputs on criteria like relevance, coherence, and overall quality. It's particularly important for open-ended tasks.|

* [A Survey on Evaluation of Large Language Models](https://arxiv.org/abs/2307.03109)
* [LMSYS Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

### Quantization

|Base techniques|[Quantization](https://github.com/AlleninTaipei/Artificial-Intelligence-Introduction-for-Beginners/blob/main/AI%20Introduction.md#quantization) reduces model size and inference speed by converting high-precision floating-point weights to lower-precision formats.|
|-|-|
|Int8 quantization|Converts 32-bit floats to 8-bit integers|
|Int4 quantization|Uses 4-bit integers for further compression|
|Mixed-precision|Combines different precisions for different layers|
|GGUF and [llama.cpp](https://github.com/ggerganov/llama.cpp)|GGUF (GPT-Generated Unified Format) is a file format for quantized models|
||llama.cpp is an efficient C++ implementation for running LLMs on CPUs|
||Supports various quantization levels (4-bit, 5-bit, 8-bit)|
||Optimized for inference on consumer hardware|
|[GPTQ](https://arxiv.org/abs/2210.17323) and [EXL2](https://github.com/turboderp/exllamav2)|GPTQ (GPT Quantization) is a quantization method for LLMs|
||Uses vector-wise quantization and second-order information|
||EXL2 is an optimized CUDA implementation of GPTQ|
||Allows for 3-bit and 4-bit quantization with minimal accuracy loss|
|[AWQ (Activation-aware Weight Quantization)](https://arxiv.org/abs/2306.00978)|Considers activation distributions during quantization|
||Adaptively chooses quantization parameters for each layer|
||Can achieve high compression rates (2-4 bit) with less accuracy loss|
||Often used in conjunction with other techniques|

* [Introduction to Weight Quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html)
* [Quantize Llama models with GGUF and llama.cpp](https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html)
* [4-bit LLM Quantization with GPTQ](https://mlabonne.github.io/blog/posts/4_bit_Quantization_with_GPTQ.html)
* [ExLlamaV2: The Fastest Library to Run LLMs](https://mlabonne.github.io/blog/posts/ExLlamaV2_The_Fastest_Library_to_Run%C2%A0LLMs.html)

### New Trends

|New Trends||
|-|-|
|**Positional embeddings**|These are used in transformer models to provide information about the relative or absolute position of tokens in a sequence. They're crucial for models to understand the order and structure of input data.<br>[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)<br>[YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)<br>[Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)|
|**Model merging**|This technique involves combining multiple trained models to create a new model that potentially inherits the strengths of its "parent" models. It's an active area of research for improving model performance and capabilities.<br>[Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099)<br>[Merge Large Language Models](https://mlabonne.github.io/blog/posts/2024-01-08_Merge_LLMs_with_mergekit.html) with [mergekit](https://github.com/arcee-ai/mergekit)|
|[Mixture of Experts (MoE)](https://arxiv.org/abs/2401.04088)|This approach uses multiple "expert" neural networks, each specialized for different tasks or data types, with a gating mechanism to route inputs to the most appropriate expert. MoE can improve efficiency and performance, especially for large-scale models.<br>[phixtral-2x2_8](https://huggingface.co/mlabonne/phixtral-2x2_8)|
|**Multimodal models**|[CLIP: Connecting text and images](https://openai.com/index/clip/)<br>[Stable Diffusion 3 Medium](https://stability.ai/stable-image)<br>[LLaVA: Large Language and Vision Assistant](https://llava-vl.github.io/)|

* [Mixture of Experts Explained](https://huggingface.co/blog/moe)
* [Multimodality and Large Multimodal Models (LMMs)](https://huyenchip.com/2023/10/10/multimodal.html)

---

## LLM Fundamentals Roadmap

### Mathematics for Machine Learning

|Linear Algebra||
|-|-|
|Vectors|Ordered lists of numbers, representing points in space.|
|Matrices|Rectangular arrays of numbers, representing transformations or datasets.|
|Addition/Subtraction|Element-wise operations on vectors/matrices.|
|Scalar Multiplication|Multiplying every element by a scalar.|
|Matrix Multiplication|Combining transformations or datasets.|
|Dot Product|Measure of similarity between two vectors.|
|Cross Product|Produces a vector perpendicular to two 3D vectors.|
|Determinant|Scalar value representing the volume scaling factor of the transformation.|
|Inverse|Matrix that reverses the transformation of the original matrix.|
|Transpose|Flipping a matrix over its diagonal.|
|Eigenvalues|Scalars indicating the factor by which the eigenvector is scaled during the transformation.|
|Eigenvectors|Vectors unchanged in direction during the transformation.|
|Singular Value Decomposition (SVD)|Factorizes a matrix into three other matrices, revealing its structure.|
|Principal Component Analysis (PCA)|Dimensionality reduction technique using eigenvalues and eigenvectors.|

|Calculus||
|-|-|
|Differential Calculus|Derivatives: Measure the rate of change of a function.|
||Partial Derivatives: Derivatives with respect to one variable in multivariable functions.|
||Gradient: Vector of partial derivatives, indicating the direction of the steepest ascent.|
|Integral Calculus|Integrals: Measure the area under a curve, representing accumulation of quantities.|
||Definite Integrals: Evaluate the net area between two points.|
||Indefinite Integrals: General form representing antiderivatives.|
|Optimization|Gradient Descent: Iterative method to find the minimum of a function using gradients.|
||Convex Functions: Functions where any local minimum is a global minimum.|

|Probability||
|-|-|
|Basic Concepts|Random Variables: Variables that take on different values randomly.|
||Probability Distributions: Functions that describe the likelihood of different outcomes.|
||Expected Value: Mean of a random variable's probability distribution.|
||Variance: Measure of the spread of a distribution.|
|Distributions|Discrete Distributions: Such as Binomial and Poisson distributions.|
||Continuous Distributions: Such as Normal (Gaussian) and Exponential distributions.|
|Bayes' Theorem|Describes the probability of an event based on prior knowledge of conditions related to the event.|
|Markov Chains|Models where the next state depends only on the current state (memoryless property).|

|Statistics||
|-|-|
|Descriptive Statistics|Mean, Median, Mode: Measures of central tendency.|
||Standard Deviation, Variance: Measures of spread.|
|Inferential Statistics|Hypothesis Testing: Methods to test assumptions about a population.|
||Confidence Intervals: Range of values likely to contain the population parameter.|
|Regression Analysis|Linear Regression: Modeling the relationship between a dependent variable and one or more independent variables.|
||Logistic Regression: Modeling the probability of a binary outcome.|
|Clustering|K-Means: Partitioning data into K clusters.|
||Hierarchical Clustering: Building a hierarchy of clusters.|

### Python for Machine Learning

* [Real Python Tutorials](https://realpython.com/)
* [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

### Machine Learning

* [Supervised Learning, Unsupervised Learning](https://github.com/AlleninTaipei/Artificial-Intelligence-Introduction-for-Beginners/blob/main/AI%20Introduction.md#machine-learning)

|Model evaluation|It is crucial to assess the performance of a machine learning model. Proper evaluation ensures that the model generalizes well to unseen data and meets the desired criteria.|
|-|-|
|Metrics|Various metrics are used depending on the type of task.|
||Classification: Accuracy, precision, recall, F1 score, ROC-AUC.|
||Regression: Mean absolute error (MAE), mean squared error (MSE), R-squared.|
|Validation Techniques|Train-Test Split: Dividing the dataset into training and testing sets.|
||Cross-Validation: Dividing the data into k subsets (folds) and training/testing the model k times with a different fold as the test set each time. Common techniques include k-fold cross-validation and stratified k-fold cross-validation.|
|Overfitting|When the model performs well on training data but poorly on unseen data due to learning noise and details of the training data.|
|Underfitting|When the model is too simple to capture the underlying pattern of the data, resulting in poor performance on both training and test data.|

### Neural Networks

|Neural Networks|Neural networks are a set of algorithms, modeled loosely after the human brain, designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling, or clustering of raw input. The basic building block of a neural network is the neuron, often referred to as a node or unit.|
|-|-|
|Neuron|A single computational unit in a neural network. It takes input, processes it, and generates an output.|
|Layer|A collection of neurons. The three primary types of layers are input, hidden, and output layers.|
|Activation Function|Determines the output of a neuron given a set of inputs. Common activation functions include sigmoid, tanh, and ReLU.|
|Weight|Parameters within the neural network that transform input data within the network's layers.|
|Bias|A parameter that allows you to shift the activation function.|

|Steps in Training|Training a neural network involves adjusting the weights and biases to minimize the error in the output. This is typically done using a process called backpropagation in conjunction with an optimization algorithm.|
|-|-|
|Forward Propagation|Input data is passed through the network, layer by layer, until it reaches the output layer.|
|Loss Function|Measures the difference between the network's output and the true value. Common loss functions include Mean Squared Error (MSE) for regression tasks and Cross-Entropy Loss for classification tasks.|
|Backpropagation|The process of calculating the gradient of the loss function with respect to each weight by the chain rule, starting from the output layer and moving backward through the network.|
|Optimization Algorithm|Updates the weights and biases to minimize the loss function. Common optimization algorithms include Gradient Descent, Adam, and RMSprop.

|Techniques to Prevent Overfitting|Overfitting occurs when a model learns the training data too well, including its noise and outliers, leading to poor performance on unseen data.|
|-|-|
|Regularization|Techniques like L1 and L2 regularization add a penalty for large weights to the loss function.|
|Dropout|Randomly sets a fraction of input units to zero at each update during training time, which helps prevent neurons from co-adapting too much.|
|Early Stopping|Stops training when the performance on a validation set starts to degrade, indicating that the model is beginning to overfit.|
|Cross-Validation|Splitting the data into multiple training and validation sets to ensure the model generalizes well.|

#### Implementing a Multilayer Perceptron (MLP)

* A Multilayer Perceptron is a class of feedforward artificial neural network (ANN) that consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer.

* Here’s how you can implement a simple Multi-Layer Perceptron (MLP) using PyTorch

        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        import numpy as np

        # Assuming you have your data loaded in variables `X` and `y`
        # For demonstration, let's create dummy data
        X = np.random.rand(1000, 784).astype(np.float32)
        y = np.random.randint(0, 10, 1000).astype(np.int64)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)
        X_test = torch.tensor(X_test)
        y_test = torch.tensor(y_test)

        # Create DataLoader for training and test sets
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Define the neural network architecture
        class MLP(nn.Module):
            def __init__(self):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(784, 64)
                self.fc2 = nn.Linear(64, 10)
                self.relu = nn.ReLU()
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.softmax(self.fc2(x))
                return x

        # Instantiate the model, define the loss function and the optimizer
        model = MLP()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training the model
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        # Evaluating the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Test Accuracy: {correct / total:.4f}')


### Natural Language Processing

* Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful.

|Text Preprocessing|Text preprocessing involves transforming raw text into a format that can be effectively used by machine learning models.|
|-|-|
|Tokenization|Splitting text into smaller units like words or sentences.|
|Lowercasing|Converting all characters in the text to lowercase to maintain uniformity.|
|Removing Punctuation|Stripping punctuation marks to avoid treating punctuations as separate words.|
|Stop Words Removal|Removing common words (like 'and', 'the', etc.) that do not contribute much to the meaning.|
|Stemming|Reducing words to their base or root form (e.g., 'running' to 'run').|
|Lemmatization|Reducing words to their base or dictionary form (e.g., 'better' to 'good').|

    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Example text
    text = "Natural Language Processing is fascinating. It's exciting to learn NLP!"

    # Tokenization
    tokens = word_tokenize(text.lower())

    # Removing punctuation
    tokens = [word for word in tokens if word.isalnum()]

    # Removing stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    print("Original Tokens:", tokens)
    print("Stemmed Tokens:", stemmed_tokens)
    print("Lemmatized Tokens:", lemmatized_tokens)

|Feature Extraction Techniques|Feature extraction involves transforming text data into numerical vectors that can be fed into machine learning models.|
|-|-|
|Bag of Words (BoW)|Represents text as a set of word frequencies.|
|TF-IDF (Term Frequency-Inverse Document Frequency)|Adjusts the word frequencies by how commonly the words appear across multiple documents.|
|Word Embeddings|Maps words to vectors of real numbers in a high-dimensional space.<br>Word embeddings are dense vector representations of words that capture their meanings, semantic relationships, and syntactic roles. Popular word embedding techniques include Word2Vec, GloVe, and FastText.|

    # Using Scikit-learn to extract features:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    # Example text data
    documents = [
        "Natural Language Processing is fascinating.",
        "It's exciting to learn NLP!",
        "Language processing includes many tasks."
    ]

    # Bag of Words
    vectorizer = CountVectorizer()
    X_bow = vectorizer.fit_transform(documents)
    print("Bag of Words:\n", X_bow.toarray())

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(documents)
    print("TF-IDF:\n", X_tfidf.toarray())

####

    #Using Gensim to load pre-trained word embeddings:
    import gensim.downloader as api

    # Load pre-trained GloVe embeddings
    word2vec_model = api.load("glove-wiki-gigaword-100")

    # Get the vector for a word
    vector = word2vec_model['language']

    # Find most similar words
    similar_words = word2vec_model.most_similar('language', topn=5)
    print("Most similar words to 'language':", similar_words)

|Recurrent Neural Networks (RNNs)|RNNs are a class of neural networks that are particularly good at modeling sequence data, such as time series or natural language, by maintaining a memory of previous inputs through their recurrent connections.|
|-|-|
|Standard RNNs|Have issues with long-term dependencies due to vanishing and exploding gradients.|
|Long Short-Term Memory (LSTM)|Designed to handle long-term dependencies by introducing a cell state and gates to control information flow.|
|Gated Recurrent Unit (GRU)|A simplified version of LSTM with fewer gates, offering similar performance.|

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    # Example text data
    texts = [
        "I love machine learning!",
        "Natural language processing is a fascinating field.",
        "Deep learning models are very powerful."
    ]
    labels = [1, 1, 0]  # Example labels

    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=10)

    # Convert to PyTorch tensors
    X = torch.tensor(padded_sequences, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32)

    # Dataset and DataLoader
    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels
    
        def __len__(self):
            return len(self.texts)
    
        def __getitem__(self, idx):
            return self.texts[idx], self.labels[idx]

    dataset = TextDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Define the model
    class LSTMModel(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()
    
        def forward(self, x):
            x = self.embedding(x)
            _, (hn, _) = self.lstm(x)
            out = self.fc(hn[-1])
            out = self.sigmoid(out)
            return out

    # Instantiate the model
    model = LSTMModel(vocab_size=10000, embed_size=64, hidden_size=64, output_size=1)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        for texts, labels in dataloader:
            outputs = model(texts)
            loss = criterion(outputs, labels.unsqueeze(1))
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    # Example prediction
    with torch.no_grad():
        predictions = model(X)
        print("Predictions:", predictions)

