# Agentic-RAG-ChatBot-with-LangGraph-and-Streamlit-for-SDLC-and-Azure-Cloud-QA

An intelligent, multi-domain chatbot powered by Agentic Retrieval-Augmented Generation (RAG) and LangGraph, designed to answer technical questions across 2 domains - Software Development Life Cycle (SDLC) and Azure Cloud. This system uses LangGraph, LangChain, Streamlit, and 2 dedicated ChromaDB vector stores to create a graph-driven, context-aware assistant capable of routing questions to the appropriate expert pipeline.

## Objective

This chatbot is designed to support fast-growing tech teams, especially when onboarding freshers into Software Development or Azure Cloud roles. The chatbot uses domain-specific routing to maintain high precision and separation of concerns. It enables users to quickly clarify domain-specific concepts and practices without relying on constant expert support. By handling multiple domains in one interface, it benefits cross-functional teams—developers and cloud engineers—by improving knowledge access, reducing repeated queries, and streamlining onboarding.

The system uses dedicated vector stores for each domain (SDLC and Azure Cloud), which ensures:
- More accurate, domain-relevant answers
- Faster retrieval with smaller, focused indexes
- Easy scalability to new domains
- Clean separation of knowledge areas

This makes the chatbot both practical for day-to-day use and scalable for enterprise needs.

## Architecture Overview

This chatbot uses LangGraph to construct an intelligent flow of agentic components with the following structure:

[User Question]  
    ↓  
[Router Node]  
 ├──> SDLC QA Agent  
 ├──> Azure Cloud QA Agent  
 └──> No Match → "I don't know."

### Agentic RAG Pipeline Components

1. LangGraph - Directs queries to the correct domain agent
2. RetrievalQA Chain - Handles document search and response generation
3. ChromaDB - Vector store for storing and retrieving documents
4. Sentence Transformers - For embedding documents and queries
5. LLM - Generates grounded answers based on retrieved context
6. Streamlit - used for ChatBot UI

## Deployment

This chatbot is deployed using Streamlit, offering an interactive web interface to test and interact with the system.

To launch the chatbot locally:

```bash
streamlit run main.py
```

## Tech Stack

| Component         | Technology Used                          |
|------------------|------------------------------------------|
| Language Model    | OpenAI GPT-3.5 via LangChain         |
| Retrieval Layer   | LangChain RetrievalQA       |
| Embedding Model   | Sentence Transformers (MiniLM-L6-v2)     |
| Database          | ChromaDB |
| Prompt Framework          | LangChain ChatPromptTemplate |
| Orchestration     | LangGraph (LangChain)                               |
| Frontend UI       | Streamlit                               |

## Streamlit UI Screenshots

To view the chatbot interface and how users interact with the system in real time, refer to the screenshot provided in the PDF below:

📄 [View Streamlit UI Screenshots](./output_screenshots.pdf)

This PDF showcases:

- The user input field for asking SDLC and Azure Cloud related questions
- Model-generated answers
- Clean and responsive Streamlit layout
- Real-time query response flow
  

## Library Versions

Below are the versions of the core libraries used in this project:

- langchain==0.3.25
- langchain-core==0.3.63
- langchain-community==0.3.25
- chromadb==1.0.12
- huggingface_hub==0.33.0
- PyMuPDF==1.26.1
- transformers==4.34.0
- sentence-transformers==3.4.1
- streamlit==1.45.1
- streamlit_chat==0.1.1
- bitsandbytes==0.46.0


## Contact
Anusha Chaudhuri [anusha761]
