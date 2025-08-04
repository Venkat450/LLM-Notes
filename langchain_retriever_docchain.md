# LangChain Retriever & Document Chain: Problem-Solution Story Flow

## Problem Statement

Imagine you are building an AI-powered assistant that can answer questions based on a specific web article.

For example, given this link:
```
https://www.newsweek.com/h1b-visa-immigrants-tech-jobs-impact-college-grads-2106392
```
You want to build a system where a user can ask:
> "What is the new update related to H1-B visa?"

And your system should:
1. Search through the article.
2. Find the relevant information.
3. Generate a precise answer.

But here’s the **challenge**:
- The article could be **very long**.
- LLMs have a **limited context window**.
- You can’t just dump the entire article into the LLM.
- You need an efficient way to **retrieve only the relevant parts** and let the LLM answer.

---

## Solution Flow (Step-by-Step Story)

### Step 1: **Data Ingestion**
You start by **loading the web article**.
```python
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(["https://www.newsweek.com/h1b-visa-immigrants-tech-jobs-impact-college-grads-2106392"])
docs = loader.load()
```
**Why?**
- This gives you a raw document object (`docs`) that contains the entire article text.

---

### Step 2: **Chunking the Document**
Now, you split the document into **smaller chunks**.
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
```
**Why?**
- LLMs can’t process the entire document if it exceeds token limits.
- Chunking ensures that you can search and feed only small, manageable portions to the LLM later.

---

### Step 3: **Embedding the Chunks (Vector Store)**
You convert each chunk into an **embedding (vector representation)** and store them in a vector database (FAISS).
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)
```
**Why?**
- This allows you to **perform similarity searches** to find which chunks are relevant to a query.

---

### Step 4: **Retriever (Search Interface)**
Convert the vector DB into a **Retriever interface**.
```python
retriever = db.as_retriever()
```
**Why?**
- Retriever provides a **standard interface** to fetch relevant chunks for any user query.
- This abstracts away the low-level vector search logic.

---

### Step 5: **Document Chain (Context Formatting)**
Create a **StuffDocumentsChain** to control how the retrieved chunks are formatted into a prompt.
```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context:
<context>
{context}
</context>
""")

document_chain = create_stuff_documents_chain(llm, prompt)
```
**Why?**
- Even after retrieval, you need to properly **format and inject context into a prompt**.
- StuffDocumentsChain handles this by “stuffing” all retrieved chunks into the LLM’s prompt.

---

### Step 6: **Retriever Chain (Full Pipeline)**
Now, you chain the **Retriever + Document Chain** into a single workflow.
```python
from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)
```
**Why?**
- This automates the entire flow:
  1. User inputs a query.
  2. Retriever fetches relevant chunks.
  3. Document Chain formats context.
  4. LLM generates a final answer.

---

### Step 7: **User Query Execution**
You now invoke the chain to answer questions.
```python
response = retrieval_chain.invoke({"input": "What is the new update related to H1-B visa?"})
print(response['answer'])
```
**Why?**
- This is the final step where the system dynamically pulls relevant data, builds context, and answers.

---

## Why Can't We Skip Document Chain?
- You might think: **“I got the chunks from Retriever, can’t I send them directly to the LLM?”**
  - Yes, but you’d need to **manually format them** into a proper prompt.
  - Document Chain automates this formatting.
  - For large documents, you might need advanced workflows (MapReduce, Refine) to summarize or iteratively refine answers, which Document Chains manage seamlessly.

---

## Visual Story Flow
```
Web Article → Chunking → Vector DB (FAISS) → Retriever → Document Chain (Stuff Prompt) → LLM → Answer
```

## TL;DR (The Story Recap)
- **Problem**: LLM context limit, need to pull relevant info from large documents.
- **Solution**:
  1. Load data.
  2. Chunk the document.
  3. Embed into Vector Store.
  4. Use Retriever to fetch relevant chunks.
  5. Document Chain formats context for LLM.
  6. Retrieval Chain links everything.
  7. LLM generates precise answer.

Document Chains don’t summarize on their own — they define **how and when to call the LLM for summarizing, refining, or answering** based on retrieved context.

---

## Bonus:
### Alternative Document Chains Workflows
| Document Chain Type | What It Does |
|---------------------|--------------|
| StuffDocumentsChain | Dumps all chunks into prompt |
| MapReduceDocumentsChain | Summarizes each chunk first (Map), then combines summaries (Reduce) |
| RefineDocumentsChain | Iteratively refines answer with each chunk |

---

This story-driven workflow ensures that you understand **why each step exists**, **what problem it solves**, and **how LangChain automates retrieval-augmented generation (RAG)** for LLM applications.

