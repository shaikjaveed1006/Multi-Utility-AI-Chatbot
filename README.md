🤖 Multi-Utility AI Chatbot with Production Observability

📖 Overview
An enterprise-grade conversational AI system that combines document analysis, web search, and computational capabilities with full observability and persistent memory. Built with LangGraph for complex workflows and LangSmith for production monitoring.
🎯 Key Highlights

📄 RAG Pipeline: Upload PDFs and query them with semantic search (FAISS)
🔍 Web Search: Real-time information retrieval via DuckDuckGo
🧮 Calculator: Built-in computational tools
💾 Persistent Memory: SQLite-backed conversation storage
🎨 Modern UI: Dark-themed Streamlit interface
📊 LangSmith Observability: Full request tracing and analytics
🤖 AI-Generated Titles: Automatic conversation summarization
🔄 Multi-threaded: Manage multiple conversations simultaneously



🏗️ Architecture
┌─────────────────────────────────────────────────────┐
│                Streamlit Frontend                    │
│          (Dark-Themed Modern Interface)              │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│               LangGraph Backend                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ RAG Tool │  │  Search  │  │Calculator│          │
│  └──────────┘  └──────────┘  └──────────┘          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │   LangSmith    │  ◄── Real-time tracing
            │   Monitoring   │      Token analytics
            └────────────────┘      
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│  FAISS   │  │  SQLite  │  │   Groq   │
│  Vector  │  │ Storage  │  │   LLM    │
└──────────┘  └──────────┘  └──────────┘

Tech Stack:

Framework: LangGraph, LangChain
LLM: Groq (Llama 3.3 70B) - Fast inference
Observability: LangSmith - Complete tracing
Vector DB: FAISS - Semantic search
Embeddings: HuggingFace Sentence Transformers
Frontend: Streamlit - Interactive UI
Storage: SQLite - Persistent conversations
Tools: DuckDuckGo Search, Custom Calculator