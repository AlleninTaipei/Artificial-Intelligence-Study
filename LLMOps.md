# LLMOps

Large language model operations (LLMOps) is a discipline that combines several techniques – such as **prompt engineering and management,**, **deploying LLM agents**, and **LLM observability** – to optimize language models for specific contexts and make sure they provide the expected output to users.

## Prompt Engineering

A prompt is simply the specific task a user provides to a language model, and the response is the output of the language model that accomplishes the task. A carefully crafted prompt can guide the model towards producing the desired output, while a poorly crafted prompt may result in irrelevant or nonsensical results. 

|Prompt Engineering|Prevailing Approaches|
|-|-|
|**Few-Shot Prompting**|The user provides a few examples of the task that the large language model should perform as well as a description of the task.|
|**Instructor-Based Prompting**|Base on instructing the large language model to act as a specific person while performing the desired task.|
|**Chain of Thought Prompting / CoT Prompting**|To accomplish complex tasks where the user breaks down a specific task into smaller sub-tasks and instructs the language model to perform small tasks in an incremental order in order to get to the final desired outcome.|
|**Automatic Prompt Generation**|The user simply describes the task that they want to accomplish within a few sentences and asks the language model to come up with different options. The user then searches for the best prompt.|

### Prompt Templates

With pre-amble texts that is placed right before a user’s prompt, LLM developers can standardize the output format and quality regardless of the simplicity of the prompt provided by the user.

```plaintext
prompt_template = """
I want you to act as a branding expert for new companies.
You need to come up with names to certain tech startups. Here are some examples of good company names:

- search engine, Google
- social media, Facebook
- video sharing, YouTube

The name should be short, catchy and easy to remember. What is a good name for a company that makes {product}?
"""
```

Within your LLM application, you can have different prompt templates and user inputs running continuously and it is very important to store your prompts and control their workflow. 

## LLM Agents

Collect relevant data, utilize various methods to process it, and tweak the LLM to ensure that it can deliver optimal responses within your business context. 

LLM agents assists users in generating responses quickly by creating a sequence of related prompts and answers in a logical order.

[LangChain](https://python.langchain.com/v0.2/docs/introduction/) is a framework for developing applications powered by large language models (LLMs).

|LLM application lifecycle|LangChain|
|-|-|
|**Development**|Build your applications using LangChain's open-source building blocks, components, and third-party integrations. Use LangGraph to build stateful agents with first-class streaming and human-in-the-loop support.|
|**Productionization**|Use LangSmith to inspect, monitor and evaluate your chains, so that you can continuously optimize and deploy with confidence.|
|**Deployment**|Turn your LangGraph applications into production-ready APIs and Assistants with LangGraph Cloud.|

General purpose agents that are commonly used in LangChain and LlamaIndex chains are designed for straightforward, end-to-end tasks. They take a chain of tasks, generate prompts and responses using large language models, retrieve documents, leverage tools and finally at the end of the chain, they return the result. This makes them fast and easy to deploy, but limited in scope. Complex agents like BabyAGI and Voyager incorporate more components for long-running tasks. Their architectures may include modules for memory, planning, learning over time, and iterative processing. 

You may not need to use an agent if the end doesn’t justify the means. If you do decide to use an agent, the next step is to consider if you need your objective to be a single straightforward task, or if you have a task that you would want to continually iterate over. If you are planning on using an agent for the long run you may even want to invest in an architecture that can keep skills in order to leverage them over time as it continues to learn. 

If you are looking to build and maintain an agent in production, either through monitoring its chain of thought traces or by the performance of its final iteration, the field of LLMOps is testing out agent capabilities and trying to further our own understanding of agent architectures. Currently, an agent does not come as a one-size-fits-all solution for a use case given they are complex, largely untested and computationally expensive at scale. That said, agents don’t seem to be going anywhere for now and are quickly becoming agents of change within the generative AI community.

## LLM Observability

Observability for LLMs (Large Language Models) refers to the tools and processes used to monitor and understand how LLM applications perform in real time. This involves tracking all the input prompts, templates, and the responses generated by the model. It helps prompt engineers diagnose issues quickly, identify the root causes of negative feedback, and optimize the prompts and model behavior accordingly.

|Key Components|LLM Observability|
|-|-|
|**Prompt and Response Monitoring**|All the prompts sent to the LLM and the corresponding responses are monitored. This data allows prompt engineers to see the LLM's output in different contexts and scenarios.|
|**Data Collected by an Observability System**|Prompt and Response: The actual input and output data.|
||Prompt and Response Embedding: Embeddings are vector representations of the prompts and responses.|
||Prompt Templates: The templates used for generating prompts.|
||Prompt Token Length: The number of tokens in a prompt.|
||Response Token Length: The number of tokens in the response.|
||Step in Conversation: The order of the prompt/response within a conversation.|
||Conversation ID: An identifier for the conversation.|
||Structured Metadata: Additional data tagged with predictions, like user feedback.|
||Embedded Metadata: Metadata integrated into the prompts or responses.|
||Embeddings: Embeddings are internal representations of data, capturing the essence of prompts or responses in vector form. Even if teams don't have direct access to model embeddings (like those from GPT-4), they can generate them using other models (e.g., GPT-J or BERT). These embeddings can be tracked to monitor changes in the model's "thinking" or decision-making process.|
|**Clustering for Issue Detection**|By clustering prompts and responses based on their embeddings, teams can identify patterns and commonalities among problematic responses. This helps in quickly isolating and addressing issues.|
|**Troubleshooting Workflow**|**Identifying Problematic Clusters**: Observability tools can highlight clusters of responses that have been flagged as problematic, either due to negative feedback or deviations from expected behavior.|
||**Cluster Analysis and Comparison**: By analyzing clusters and comparing them against baseline datasets, engineers can identify specific issues within the clusters.|
||**Interactive Analysis**: Tools enable engineers to perform exploratory data analysis (EDA) to understand and fix issues.|
|**Latency and Performance Monitoring**|Observability solutions can also track performance metrics like API latency. This allows teams to identify prompt/response pairs that cause delays and investigate them further.|

||How LLM Observability Helps|
|-|-|
|**Real-Time Monitoring:**|Engineers get instant insights into how the LLM is performing, allowing for quick adjustments and fixes.|
|**Root Cause Analysis**|By understanding the structure and patterns of prompts and responses, engineers can pinpoint the exact cause of any issues.|
|**Optimization and Fine-Tuning**|Continuous monitoring and analysis allow for ongoing optimization of the LLM and its prompts, leading to improved performance and user satisfaction.|
|**Feedback Integration**|User feedback, such as ratings or comments, can be integrated into the observability process to ensure that the LLM's outputs align with user expectations.|

||Example of a Problematic Cluster Workflow|
|-|-|
|**Detecting Issues**|The observability system detects a cluster of responses that users have rated poorly.|
|**Analyzing the Cluster**|Engineers use the observability tool to analyze the cluster, looking for common patterns or issues within the responses.|
|**Fixing the Problem**|Engineers iterate on prompt engineering or fine-tuning to address the issues, ensuring that future responses are more accurate and satisfactory.|

---

## LLM Observability approaches

|Approaches|Methods|
|-|-|
|[LLM Evaluation](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLMOps.md#llm-evaluation---llm-as-a-judge)|You can collect the feedback directly from your users. This is the simplest way but can often suffer from users not being willing to provide feedback or simply forgetting to do so. The other approach is to use an LLM to evaluate the quality of the response for a particular prompt. This is more scalable and very useful but comes with typical LLM setbacks.|
|[Traces and Spans In Agentic Workflows](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLMOps.md#traces-and-span)|For more complex or agentic workflows, it may not be obvious which call in a span or which span in your trace (a run through your entire use case) is causing the problem. You may need to repeat the evaluation process on several spans before you narrow down the problem.|
|[Search and Retrieval (aka Retrieval Augmented Generation)](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLMOps.md#search-and-retrieval)|You may first embed the documents by summaries. Then at retrieval time, find the document by summary first, then get relevant chunks. Or you can embed text at the sentence level, then expand that window during LLM synthesis. Or maybe you can just embed the reference to the text or even change how you organize information on the back end. There are many possibilities to decouple embeddings from the text chunks.|
|**Fine-tuning**|Fine-tuning essentially generates a new model that is more aligned with your exact usage conditions.|
|[Prompt engineering](https://github.com/AlleninTaipei/Artificial-Intelligence-Study/blob/main/LLMOps.md#prompt-engineering)|LLMs are based on the attention mechanism, and the magic behind the attention mechanism is that it is really good at picking up relevant context. You just need to provide it. The tricky part in product applications is that you not only have to get the right prompt, you also have to try to make it concise. LLMs are priced per token. The other issue is that LLMs have a limited context window, so there is only so much context you can provide in your prompt.|

### LLM Evaluation - LLM As a Judge

LLM-assisted evaluation uses AI to evaluate AI — with one LLM evaluating the outputs of another and providing explanations.

#### What Is the Difference Between LLM Model Evaluation and LLM System Evaluation (AKA Task Evaluations)?

|Model Evaluations|Task Evaluations|
|-|-|
|overall general assessment|assessing performance of a particular task|
|look at the “general fitness” of the model (how well does it do on a variety of tasks?)|specifically designed to look at how well the model is suited for a particular application|
|someone who works out and is quite fit generally|a professional sumo wrestler in a real competition|

#### Model Evaluations

Evaluate two different open source foundation models. Testing the same dataset across the two models and seeing how their metrics, like hellaswag or mmlu, stack up.

[HellaSwag dataset](https://rowanzellers.com/hellaswag/)  consists of a collection of contexts and multiple-choice questions where each question has multiple potential completions. Only one of the completions is sensible or logically coherent, while the others are plausible but incorrect. These completions are designed to be challenging for AI models, requiring not just linguistic understanding but also common sense reasoning to choose the correct option.

[MMLU features tasks](https://paperswithcode.com/dataset/mmlu) that span multiple subjects, including science, literature, history, social science, mathematics, and professional domains like law and medicine. This diversity in subjects is intended to mimic the breadth of knowledge and understanding required by human learners, making it a good test of a model’s ability to handle multifaceted language understanding challenges

[The Hugging Face Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) is perhaps the best known place to get such model evaluations. The leaderboard tracks open source large language models and keeps track of many model evaluation metrics. This is typically a great place to start understanding the difference between open source LLMs in terms of their performance across a variety of tasks.

[The Gemini paper](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf) demonstrates that multi-modality introduces a host of other benchmarks like VQAv2, which tests the ability to understand and integrate visual information. This information goes beyond simple object recognition to interpreting actions and relationships between them.

#### Task Evaluations

Evaluate two different prompt templates on a single foundational model. Test the same dataset across the two templates and seeing how their metrics like precision and recall stack up.

**Extracting structured information**: You can look at how well the LLM extracts information. For example, you can look at completeness (is there information in the input that is not in the output?).

**Question answering**: How well does the system answer the user’s question? You can look at the accuracy, politeness, or brevity of the answer—or all of the above.

**Retrieval Augmented Generation (RAG)**: Are the retrieved documents and final answer relevant?

|Type|Description|Example Metrics|
|-|-|-|
|Diversity|Examines the versatility of foundation models in responding to different types of queries|Fluency, Perplexity, ROUGE scores|
|User Feedback|Goes beyond accuracy to look at response quality in terms of coherence and usefulness|Coherence, Quality, Relevance|
|Ground Truth-Based Metrics|Compares a RAG system’s responses to a set of predefined, correct answers|Accuracy, F1 score, Precision, Recall|
|Answer Relevance|How relevant the LLM’s response is to a given user’s query.|Binary classification (Relevant/Irrelevant)|
|QA Correctness|Based on retrieved data, is an answer to a question correct?|Binary classification (Correct/Incorrect)|
|Hallucinations|Looking at LLM hallucinations with regard to retrieved context|Binary classification (Factual/Hallucinated)|
|Toxicity|Are responses racist, biased, or toxic?|Disparity Analysis, Fairness Scoring, Binary classification (Non-Toxic/Toxic)|

#### Recap: Differences Between LLM Model Evaluations and LLM Task Evaluations

|Category|Model Evaluations|Task Evaluations|
|-|-|-|
|**Foundation of Truth**|Relies on benchmark datasets.|Relies on the golden dataset curated by internal experts and augmented with LLMs.|
|**Nature of Questions**|Involves a standardized set of questions, ensuring a broad evaluation of capabilities.|Utilizes unique, task-specific prompts, adaptable to various data scenarios, to mimic real-world scenarios.|
|**Frequency and Purpose**|Conducted as a one-off test to grade general abilities, using established benchmarks.|An iterative process, applied repeatedly for system refinement and tuning, reflecting ongoing real-world applications.|
|**Value of Explanations**|Explanations don’t typically add actionable value; focus is more on outcomes.|	Explanations provide actionable insights for improvements, focusing on understanding performance in specific contexts.|
|**Persona**|LLM researcher evaluating new models and ML practitioner selecting a model for her application.|ML practitioner throughout the lifetime of the application.|

### Traces and Span

LLM orchestration frameworks (like LangChain or LlamaIndex) are trying to enable developers with the necessary tools to build LLM applications and LLM observability is designed to manage and maintain these applications in production. This orchestration process includes many different components including: programmatic querying, retrieving contextual data from a vector database, and maintaining memory across LLM and API calls.

**Example**: Let’s give the scenario that you are a Software Engineer at an e-commerce company which recently pushed an LLM-powered chatbot into production. Your chatbot, which is used to interact with customers who have purchased from your company’s website, uses a search and retrieval system to create responses for the customer

**Implmentation**: A document is broken into chunks, these chunks are embedded into a vector store, and the search and retrieval process pulls on this context to shape LLM

**Definition of LLM Observability Terms for Reference**

|Term|LLM Observability Definition|
|-|-|
|**Traces**|Traces represent a single invocation of an LLM application. For example, when a chain is run or a query engine is queried, that is a trace. Another way to think of traces is as a sequence of spans tied together by a trace ID|
|**Spans**|Spans are units of execution that have inputs and outputs that a builder of an LLM application may care to evaluate. There are different kinds of spans, including chain spans, LLM spans, and embedding spans, that are differentiated by various kinds of attributes. For example LLM span type: Attributes = Temperature, Provider, Max Tokens, …|
|**Tools**|Tool as the defining feature of an agent, an arbitrary function (e.g., a calculator, a piece of code to make an API call, a tool to query a SQL database) that an LLM can choose to execute or not based on the input from a user or the state of an application.|
|**Parent-Child Relationships Between Spans**|Every trace has a hierarchical structure. The top span, which is the entry point, has no parent. However, as you delve deeper into the system’s operations, you’ll find child spans that are initiated by their parent spans.|
|**Conversations**|Conversations are a series of traces and spans tied together by a Conversation ID. These occur across traces without any parallel operations and contain a single back and forth conversation between the LLM and a given user.|

When you execute a LLM run, the process of interacting with your selected LLM is documented in a callback system by a trace. In this trace a span can refer to any unit of execution, you may annotate a span with a specific name (agent, LLM, tool, embedding) or a general term like a chain (which can refer to any process that doesn’t have its own span kind).

### Search and Retrieval

Creating chatbots that are customized for your specific business needs involves **leveraging your unique knowledge base.**

Retrieval augmented generation is a technique where the content produced by large language models (LLMs) is enriched or *augmented* through the addition of relevant material *retrieved* from external sources.

If you are building a RAG system, you are adding recent knowledge to a LLM application system in hopes that retrieved relevant knowledge will increase factuality and decrease model hallucinations in query responses. ​

|When to avoid adding RAG|Reasons|
|-|-|
|**Reduce cost and risk**|you are trying to optimize performance|
|**Using proprietary data**|If the LLM application you are using does not require secure data to generate responses you can prompt the LLM directly and use additional tools and agents to keep responses relevant.|
|**New to prompt engineering**|We recommend experimenting with prompt templates and prompt engineering to ask better questions to your LLMs and to best structure your outputs. However, prompts do not add additional context to your user’s queries.|
|**Fine-tuning your LLM**|We recommend fine-tuning your model to get better at specific tasks by providing your LLM explicit examples. Fine-tuning should be used after experimenting with performance improvements made by prompt engineering and by adding relevant content via RAG. This is due to the speed and cost of iteration, keeping retrieval indices up-to-date is more efficient than continuously fine-tuning and retraining LLMs.|

|components of a RAG system|Detail|
|-|-|
|**Retrieval Engine**|This is the first step in the RAG process. It involves searching through a vast database of information to find relevant data that corresponds to the input query. This engine uses sophisticated algorithms to ensure the data retrieved is the most relevant and up-to-date.|
|**Augmentation Engine**|Once the relevant data is retrieved, the augmentation engine comes into play. It integrates the retrieved data with the input query, enhancing the context and providing a more informed base for generating responses.|
|**Generation Engine**|This is where the actual response is formulated. Using the augmented input, the generation engine, typically a sophisticated language model, creates a coherent and contextually relevant response. This response is not just based on the model’s preexisting knowledge but is enhanced by the external data sourced by the retrieval engine.|

|successful RAG applications|Detail|
|-|-|
|**Data Indexing**|Before RAG can retrieve information, the data must be aggregated and organized in an index. This index acts as a reference point for the retrieval engine.|
|**Input Query Processing**|The user’s input is processed and understood by the system, forming the basis of the search query for the retrieval engine.|
|**Search and Ranking**|The retrieval engine searches the indexed data and ranks the results in terms of relevance to the input query.|
|**Prompt Augmentation**|The most relevant information is then combined with the original query. This augmented prompt serves as a richer source for response generation.
|**Response Generation**|Finally, the generation engine uses this augmented prompt to create an informed and contextually accurate response.|
|**Evaluation**|A critical step in any pipeline is checking how effective it is relative to other strategies, or when you make changes. Evaluation provides objective measures on accuracy, faithfulness, and speed of responses.|

|Scenario|AI Chatbot in Customer Service|
|-|-|
|**Customer Query Processing**|When a customer asks a question, such as “What are the latest updates to your smartwatch series?”, the chatbot processes this input to understand the query’s context.|
|**Retrieval from Knowledge Base**|The retrieval engine then searches the company’s up-to-date product database for information relevant to the latest smartwatch updates.|
|**Augmenting the Query**|The retrieved information about the smartwatch updates is combined with the original query, enhancing the context for the chatbot’s response.|
|**Generating an Informed Response**|The chatbot, using the RAG’s generation engine, formulates a response that not only answers the question based on its internal knowledge base but also includes the latest information retrieved, such as new features or pricing.|

## Conclusion

Overall, LLM observability is crucial for maintaining the quality and reliability of LLM applications, especially as they scale and handle more complex tasks. By providing a comprehensive view of the model's performance, observability tools help ensure that the LLM meets the needs and expectations of users and stakeholders.

LLMOps (Large Language Model Operations), developers can optimize their LLMs to handle specific tasks, efficiently manage prompts, and monitor model performance in real-time. As the adoption of LLMs continues to expand, LLM observability allows for fine-tuning and iterative prompt engineering workflows. By identifying problematic clusters of responses, developers can refine their prompt engineering techniques or fine-tune the model to enhance its performance. This iterative process ensures continuous improvement of the LLM application, leading to a better end-user experience.

* Source
[A Developer’s Guide To LLMOps (Large Language Model Operations): Operationalizing LLMs](https://arize.com/blog-course/llmops-operationalizing-llms-at-scale/)