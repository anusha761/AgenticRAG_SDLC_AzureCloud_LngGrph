import streamlit as st
from typing import Optional, Literal, TypedDict
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langgraph.graph import StateGraph, END



# --- Model Loading and Caching---
@st.cache_resource
def init_system():
    # Load embedding model
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load vectorstores
    sdlc_db = Chroma(persist_directory="chroma_db_sdlc_store", embedding_function=embedding)
    cloud_db = Chroma(persist_directory="chroma_db_cloud_store", embedding_function=embedding)

    # Load LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Prompt template
    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an expert in {domain}. Use the provided context to answer the user's question concisely within 10 sentences. "
            "If the question is unrelated to {domain}, say 'I don't know'."
        ),
        HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion:\n{question}")
    ])

    # Build QA chains
    sdlc_qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=sdlc_db.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt.partial(domain="Software Development Life Cycle (SDLC)")}
    )

    cloud_qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=cloud_db.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt.partial(domain="Azure Cloud Computing")}
    )

    return llm, sdlc_qa, cloud_qa



# --- LangGraph Setup - cached ---
@st.cache_resource
def get_graph(_llm, _sdlc_qa, _cloud_qa):
    class AgentState(TypedDict):
        question: str
        route: Optional[Literal["sdlc", "cloud", "none"]]
        answer: Optional[str]

    def route_question(state: AgentState) -> AgentState:
        route_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert classifier. Human will ask you a question. If the question is related to SDLC respond with 'sdlc'. "
                       "If the question is related to Azure Cloud Computing, respond with 'cloud'. Otherwise, respond with 'none'. Respond with only ONE word."),
            ("human", "Question: {question}")
        ])
        chain = route_prompt | _llm
        route = chain.invoke({"question": state["question"]}).content.strip().lower()
        return {**state, "route": route}

    def sdlc_node(state: AgentState) -> AgentState:
        result = _sdlc_qa.invoke({"query": state["question"]})
        return {**state, "answer": result["result"]}

    def cloud_node(state: AgentState) -> AgentState:
        result = _cloud_qa.invoke({"query": state["question"]})
        return {**state, "answer": result["result"]}

    def no_match_node(state: AgentState) -> AgentState:
        return {**state, "answer": "I don't know."}

    graph = StateGraph(AgentState)
    graph.add_node("router", route_question)
    graph.add_node("sdlc", sdlc_node)
    graph.add_node("cloud", cloud_node)
    graph.add_node("none", no_match_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {"sdlc": "sdlc", "cloud": "cloud", "none": "none"}
    )

    graph.add_edge("sdlc", END)
    graph.add_edge("cloud", END)
    graph.add_edge("none", END)

    return graph.compile()




# --- Streamlit UI ---
st.set_page_config(page_title="Agentic RAG : SDLC & Azure Cloud ChatBot", layout="centered")
st.title("Agentic RAG : SDLC & Azure Cloud ChatBot")

# Load
llm, sdlc_qa, cloud_qa = init_system()
runnable = get_graph(llm, sdlc_qa, cloud_qa)

# Session State for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input
question = st.chat_input("Ask a question...")

# Display chat history
for q, a in st.session_state.chat_history:
    st.chat_message("user").markdown(q)
    st.chat_message("assistant").markdown(a)

# On user input
if question:
    st.chat_message("user").markdown(question)

    # Run through LangGraph
    state = runnable.invoke({"question": question})
    answer = state["answer"]

    # Display and store
    st.chat_message("assistant").markdown(answer)
    st.session_state.chat_history.append((question, answer))
