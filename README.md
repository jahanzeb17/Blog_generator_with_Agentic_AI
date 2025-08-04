# Agentic AI Blog Writer using LangGraph & Groq (LLaMA3)

This project demonstrates how to build an **Agentic AI workflow** using `LangGraph`, powered by Groq’s ultra-fast inference of `LLaMA3-8B`. From a single topic, the system generates a polished blog post through three agentic phases: outline generation, refinement, and content creation.

It showcases how agents can work in a modular, stateful pipeline—ideal for complex tasks like writing, research, and content automation.

---

## Overview

This Agentic AI system performs the following stages:

1. **📝 Outline Generation** – Generates an initial blog outline based on the given topic.
2. **🧹 Outline Refinement** – Revises and improves the outline to meet editorial quality.
3. **🧾 Blog Content Generation** – Creates a complete, well-structured blog post in Markdown format.

Each stage is implemented as a node in a LangGraph state machine, passing state along to the next.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Agentic state machine for LLM workflows |
| [LangChain](https://www.langchain.com/) | LLM orchestration and component abstraction |
| [ChatGroq](https://console.groq.com/) | LLaMA3-8B inference via Groq API |
| `dotenv` | Secure API key management |
| Python | Core programming language |

---

## 📁 Project Structure

