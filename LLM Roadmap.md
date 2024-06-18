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
|LLM Engineer Roadmap|Running LLMs|LLM APIs, Open-source LLMs, Prompt engineering, Structuring outputs|
||Building a Vector Storage|Ingesting documents, Splitting documents, Embedding models|
||Retrieval Augmented Generation|Orchestrators, Retrievers, Memory, Evaluation|
||Advanced RAG|Query construction, Agents and tools, Post-processing, Program LLMs|
||Inference Optimization|Flash Attention, Key-value cache, Speculative decoding|
||Deploying LLMs|Local deployment, Demo deployment, Server deployment, Edge deployment|
||Securing LLMs|Prompt hacking, Backdoors, Defensive measures|

---

## LLM Fundamentals Roadmap

### Mathematics for Machine Learning

#### Linear Algebra
- **Matrices and Vectors**: Understanding the basics of matrices and vectors.
- **Matrix Operations**: Matrix multiplication, inversion, and other operations.
- **Eigenvalues and Eigenvectors**: Their significance in machine learning.
- **Singular Value Decomposition (SVD)**: Applications in dimensionality reduction.

#### Calculus
- **Differential Calculus**: Derivatives and their applications in optimization.
- **Integral Calculus**: Integration techniques and applications.
- **Multivariable Calculus**: Partial derivatives, gradients, and Hessians.

#### Probability and Statistics
- **Probability Theory**: Basic concepts, distributions, and theorems.
- **Statistical Inference**: Estimation, hypothesis testing, and confidence intervals.
- **Bayesian Statistics**: Bayes’ theorem, prior, and posterior distributions.

### Python for Machine Learning

#### Python Basics
- **Syntax and Semantics**: Variables, data types, and control structures.
- **Functions and Modules**: Defining and importing functions and modules.
- **File I/O**: Reading from and writing to files.

#### Data Science Libraries
- **NumPy**: Arrays, operations, and mathematical functions.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib and Seaborn**: Data visualization techniques.

#### Data Preprocessing
- **Data Cleaning**: Handling missing values, outliers, and duplicates.
- **Feature Engineering**: Creating and transforming features.
- **Scaling and Normalization**: Techniques for feature scaling.

### Machine Learning

#### Supervised Learning
- **Regression**: Linear and logistic regression models.
- **Classification**: Decision trees, SVMs, and k-NN.
- **Ensemble Methods**: Random forests and boosting.

#### Unsupervised Learning
- **Clustering**: K-means, hierarchical, and DBSCAN.
- **Dimensionality Reduction**: PCA and t-SNE.
- **Anomaly Detection**: Techniques and applications.

#### Model Evaluation
- **Metrics**: Accuracy, precision, recall, F1-score.
- **Cross-Validation**: Techniques for model validation.
- **Hyperparameter Tuning**: Grid search and random search.

### Neural Networks

#### Fundamentals
- **Perceptron**: Basic unit of a neural network.
- **Activation Functions**: Sigmoid, ReLU, Tanh, and others.
- **Network Architectures**: Feedforward, convolutional, and recurrent networks.

#### Training and Optimization
- **Backpropagation**: Algorithm for training neural networks.
- **Gradient Descent**: Optimization techniques (SGD, Adam).
- **Loss Functions**: MSE, cross-entropy, and others.

#### Overfitting
- **Regularization**: L1, L2 regularization.
- **Dropout**: Technique to prevent overfitting.
- **Data Augmentation**: Techniques to increase dataset size.

#### Implement an MLP
- **Architecture Design**: Designing a multi-layer perceptron.
- **Implementation**: Coding an MLP from scratch.
- **Training and Evaluation**: Training the MLP and evaluating its performance.

### Natural Language Processing

#### Text Preprocessing
- **Tokenization**: Splitting text into tokens.
- **Normalization**: Lowercasing, stemming, and lemmatization.
- **Stopwords Removal**: Eliminating common stopwords.

#### Feature Extraction Techniques
- **Bag-of-Words**: Representing text data as vectors.
- **TF-IDF**: Term Frequency-Inverse Document Frequency.
- **N-grams**: Capturing sequences of words.

#### Word Embeddings
- **Word2Vec**: Skip-gram and CBOW models.
- **GloVe**: Global Vectors for Word Representation.
- **FastText**: Subword information in embeddings.

#### Recurrent Neural Networks
- **RNN Basics**: Understanding the structure of RNNs.
- **LSTM and GRU**: Advanced RNN architectures.
- **Seq2Seq Models**: Sequence-to-sequence learning.

---

## LLM Scientist Roadmap

### The LLM Architecture

#### High-level View
- **Overview**: Understanding the components of an LLM.
- **Workflow**: Data input, model processing, and output generation.

#### Tokenization
- **Token Types**: Subword tokens, byte-pair encoding (BPE).
- **Tokenizer Implementation**: Building and using tokenizers.
- **Vocabulary Management**: Creating and managing vocabulary.

#### Attention Mechanisms
- **Self-Attention**: Concept and implementation.
- **Multi-Head Attention**: Enhancing attention mechanisms.
- **Transformer Architecture**: Understanding Transformers.

#### Text Generation
- **Decoding Strategies**: Greedy, beam search, and sampling.
- **Controlling Output**: Temperature and top-k/top-p sampling.
- **Applications**: Practical use-cases of text generation.

### Building an Instruction Dataset

#### Alpaca-like Dataset
- **Data Collection**: Gathering data for instruction-based tasks.
- **Annotation**: Techniques for annotating data.
- **Quality Control**: Ensuring data quality and consistency.

#### Advanced Techniques
- **Data Augmentation**: Methods to expand the dataset.
- **Synthetic Data**: Generating data using models.
- **Active Learning**: Iteratively improving the dataset.

#### Filtering Data
- **Preprocessing**: Cleaning and preparing data.
- **Noise Reduction**: Techniques to filter out noisy data.
- **Balancing**: Ensuring dataset balance across classes.

#### Prompt Templates
- **Design**: Creating effective prompt templates.
- **Implementation**: Applying templates in training.
- **Evaluation**: Assessing the effectiveness of prompts.

### Pre-training Models

#### Data Pipeline
- **Data Collection**: Gathering large-scale datasets.
- **Preprocessing**: Cleaning and formatting data.
- **Sharding**: Managing large datasets in chunks.

#### Causal Language Modeling
- **Objective**: Understanding the causal language modeling objective.
- **Implementation**: Training models on causal objectives.
- **Applications**: Use-cases of causal language models.

#### Scaling Laws
- **Model Scaling**: Principles of scaling model size.
- **Data Scaling**: Impact of data size on performance.
- **Compute Scaling**: Balancing compute resources and efficiency.

#### High-Performance Computing
- **Infrastructure**: Setting up HPC environments.
- **Parallelism**: Data and model parallelism techniques.
- **Optimization**: Efficient use of resources.

### Supervised Fine-Tuning

#### Full Fine-Tuning
- **Process**: Steps for fine-tuning LLMs.
- **Techniques**: Best practices for effective fine-tuning.
- **Evaluation**: Assessing the performance of fine-tuned models.

#### LoRA
- **Concept**: Low-Rank Adaptation for efficient fine-tuning.
- **Implementation**: Applying LoRA to LLMs.
- **Benefits**: Advantages of using LoRA.

#### QLoRA
- **Quantized LoRA**: Combining quantization with LoRA.
- **Implementation**: Techniques for quantized fine-tuning.
- **Performance**: Evaluating quantized models.

#### DeepSpeed
- **Overview**: Introduction to DeepSpeed library.
- **Features**: Key features and capabilities.
- **Integration**: Using DeepSpeed for model training.

### Preference Alignment

#### Preference Datasets
- **Collection**: Gathering data reflecting user preferences.
- **Annotation**: Techniques for annotating preference data.
- **Quality Control**: Ensuring high-quality preference data.

#### Proximal Policy Optimization
- **Concept**: Introduction to PPO.
- **Implementation**: Applying PPO to preference alignment.
- **Evaluation**: Assessing the effectiveness of PPO.

#### Direct Preference Optimization
- **Overview**: Direct methods for preference optimization.
- **Techniques**: Implementing direct preference optimization.
- **Applications**: Practical use-cases.

### Evaluation

#### Traditional Metrics
- **Accuracy**: Measuring model accuracy.
- **Perplexity**: Understanding and calculating perplexity.
- **BLEU and ROUGE**: Metrics for language generation tasks.

#### General Benchmarks
- **Benchmarks**: Common benchmarks for LLMs.
- **Evaluation**: Techniques for benchmark evaluation.
- **Comparison**: Comparing models using benchmarks.

#### Task-specific Benchmarks
- **Custom Benchmarks**: Creating task-specific benchmarks.
- **Evaluation**: Techniques for task-specific evaluation.
- **Applications**: Practical examples of custom benchmarks.

#### Human Evaluation
- **Design**: Designing human evaluation studies.
- **Implementation**: Conducting human evaluations.
- **Analysis**: Analyzing human evaluation results.

### Quantization

#### Base Techniques
- **Quantization**: Introduction to model quantization.
- **Techniques**: Common quantization techniques.
- **Benefits**: Advantages of quantization.

#### GGUF and llama.cpp
- **GGUF**: Understanding GGUF quantization.
- **llama.cpp**: Introduction and applications.
- **Integration**: Using GGUF and llama.cpp in practice.

#### GPTQ and EXL2
- **GPTQ**: Overview and implementation.
- **EXL2**: Introduction and applications.
- **Comparison**: Comparing GPTQ and EXL2 techniques.

#### AWQ
- **Introduction**: Understanding AWQ quantization.
- **Techniques**: Implementing AWQ.

---

## LLM Engineer Roadmap

### Running LLMs

#### LLM APIs
- **Overview**: Introduction to LLM APIs.
- **Usage**: How to use various LLM APIs.
- **Best Practices**: Efficient and effective API usage.

#### Open-source LLMs
- **Exploration**: Overview of available open-source LLMs.
- **Implementation**: Setting up and running open-source models.
- **Comparison**: Comparing different open-source LLMs.

#### Prompt Engineering
- **Concepts**: Basics of prompt engineering.
- **Techniques**: Crafting effective prompts.
- **Applications**: Practical use-cases of prompt engineering.

#### Structuring Outputs
- **Formatting**: Techniques for structuring model outputs.
- **Templates**: Using templates for consistent outputs.
- **Post-processing**: Refining outputs for end-users.

### Building a Vector Storage

#### Ingesting Documents
- **Collection**: Methods for collecting documents.
- **Preprocessing**: Cleaning and preparing documents.
- **Storage**: Techniques for efficient document storage.

#### Splitting Documents
- **Techniques**: Methods for splitting documents into chunks.
- **Tools**: Tools and libraries for document splitting.
- **Applications**: Practical use-cases.

#### Embedding Models
- **Overview**: Introduction to embedding models.
- **Implementation**: Using embedding models for vector representation.
- **Evaluation**: Assessing the quality of embeddings.

### Retrieval Augmented Generation

#### Orchestrators
- **Concepts**: Introduction to RAG orchestrators.
- **Implementation**: Setting up RAG orchestrators.
- **Evaluation**: Assessing RAG performance.

#### Retrievers
- **Techniques**: Methods for retrieving relevant information.
- **Tools**: Tools and libraries for retrieval.
- **Optimization**: Improving retrieval efficiency.

#### Memory
- **Concepts**: Understanding memory in RAG systems.
- **Implementation**: Setting up memory components.
- **Applications**: Use-cases of memory in RAG.

#### Evaluation
- **Metrics**: Assessing the performance of RAG systems.
- **Benchmarks**: Common benchmarks for RAG.
- **Human Evaluation**: Conducting human evaluation for RAG.

### Advanced RAG

#### Query Construction
- **Techniques**: Methods for constructing effective queries.
- **Tools**: Tools for query construction.
- **Optimization**: Improving query effectiveness.

#### Agents and Tools
- **Overview**: Introduction to agents and tools in RAG.
- **Implementation**: Setting up and using agents.
- **Applications**: Practical use-cases.

#### Post-processing
- **Techniques**: Methods for refining RAG outputs.
- **Tools**: Tools for post-processing.
- **Evaluation**: Assessing the quality of post-processed outputs.

#### Program LLMs
- **Concepts**: Understanding programmatic LLMs.
- **Implementation**: Setting up programmatic LLMs.
- **Applications**: Practical use-cases.

### Inference Optimization

#### Flash Attention
- **Overview**: Introduction to flash attention.
- **Implementation**: Using flash attention in models.
- **Performance**: Evaluating the impact of flash attention.

#### Key-value Cache
- **Concepts**: Understanding key-value caching.
- **Implementation**: Setting up key-value caches.
- **Optimization**: Improving inference efficiency.

#### Speculative Decoding
- **Techniques**: Methods for speculative decoding.
- **Implementation**: Applying speculative decoding.
- **Evaluation**: Assessing the performance of speculative decoding.

### Deploying LLMs

#### Local Deployment
- **Overview**: Setting up local deployments.
- **Techniques**: Best practices for local deployment.
- **Tools**: Tools for local deployment.

#### Demo Deployment
- **Concepts**: Introduction to demo deployments.
- **Implementation**: Setting up demos.
- **Applications**: Use-cases of demo deployments.

#### Server Deployment
- **Techniques**: Methods for deploying on servers.
- **Tools**: Tools for server deployment.
- **Best Practices**: Efficient and secure server deployment.

#### Edge Deployment
- **Overview**: Introduction to edge deployments.
- **Techniques**: Best practices for edge deployment.
- **Applications**: Practical use-cases of edge deployment.

### Securing LLMs

#### Prompt Hacking
- **Concepts**: Understanding prompt hacking.
- **Defenses**: Techniques to defend against prompt hacking.
- **Tools**: Tools for securing prompts.

#### Backdoors
- **Overview**: Introduction to backdoors in LLMs.
- **Detection**: Methods for detecting backdoors.
- **Prevention**: Techniques to prevent backdoors.

#### Defensive Measures
- **Techniques**: General defensive measures for LLMs.
- **Implementation**: Applying defensive measures.
- **Evaluation**: Assessing the effectiveness of defenses.



---

# LLM Roadmap
## LLM Fundamentals Roadmap
### Mathematics for Machine Learning
#### Linear Algebra
#### Calculus
#### Probability and Statistics
### Python for Machine Learning
#### Python Basics
#### Data Science Libraries
#### Data Preprocessing
### Machine Learning
#### Supervised Learning
#### Unsupervised Learning
#### Model Evaluation
### Neural Networks
#### Fundamentals
#### Training and Optimization
#### Overfitting
#### Implement an MLP
### Natural Language Processing
#### Text Preprocessing
#### Feature Extraction Techniques
#### Word Embeddings
#### Recurrent Neural Networks

## LLM Scientist Roadmap
### The LLM architecture
#### High-level view
#### Tokenization
#### Attention mechanisms
#### Text generation
### Building an instruction dataset
#### Alpaca-like dataset
#### Advanced techniques
#### Filtering data
#### Prompt templates
### Pre-training models
#### Data pipeline 
#### Causal language modeling
#### Scaling laws
#### High-Performance Computin
### Supervised Fine-Tuning
#### Full fine-tuning
#### LoRA
#### QLoRA
#### DeepSpeed
### Preference Alignment
#### Preference datasets
#### Proximal Policy Optimization
#### Direct Preference Optimization

### Evaluation
#### Traditional metrics
#### General benchmarks
#### Task-specific benchmarks
#### Human evaluation
### Quantization
#### Base techniques
#### GGUF and llama.cpp
#### GPTQ and EXL2
#### AWQ
### New Trends
#### Positional embeddings
#### Model merging
#### Mixture of Experts
#### Multimodal models

## LLM Engineer Roadmap
### Running LLMs
#### LLM APIs
#### Open-source LLMs
#### Prompt engineering
#### Structuring outputs
### Building a Vector Storage
#### Ingesting documents
#### Splitting documents
#### Embedding models
### Retrieval Augmented Generation
#### Orchestrators
#### Retrievers
#### Memory
#### Evaluation
### Advanced RAG
#### Query construction
#### Agents and tools
#### Post-processing#
#### Program LLMs
### Inference optimization
#### Flash Attention
#### Key-value cache
#### Speculative decoding
### Deploying LLMs
#### Local deployment
#### Demo deployment
#### Server deployment
#### Edge deployment
### Securing LLMs
#### Prompt hacking
#### Backdoors
#### Defensive measures