# LangGraph Learning Journey

A structured, hands-on exploration of [LangGraph](https://github.com/langchain-ai/langgraph) — the framework for building stateful, multi-actor applications with LLMs. This repository documents my progressive study of LangGraph concepts, from foundational graph patterns to agentic systems.

Each notebook is self-contained, progressively more complex, and built with real use cases to reinforce deep understanding.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Graph-based LLM workflow orchestration |
| [LangChain](https://github.com/langchain-ai/langchain) | LLM abstractions and output parsing |
| [Groq](https://groq.com/) | Inference backend (Qwen 3 32B) |
| [Pydantic](https://docs.pydantic.dev/) | Structured output validation |
| Python `TypedDict` | Typed state management |

---

## Notebooks

| # | Notebook | Concept | Description |
|---|----------|---------|-------------|
| — | [`LangGraph_101.ipynb`](./LangGraph_101.ipynb) | Introduction | First contact with LangGraph — graph primitives, nodes, edges, and state |
| 0 | [`0_sequential_chains.ipynb`](./0_sequential_chains.ipynb) | Sequential Workflow | BMI calculator built as a linear node chain; demonstrates typed state, structured LLM output via Pydantic, and `START → END` graph wiring |
| 1 | [`1_parallel_workflow.ipynb`](./1_parallel_workflow.ipynb) | Parallel Workflow | Essay evaluation system that fans out to three independent graders (critical thinking, depth of analysis, language quality) and fans back in to produce an aggregated score |
| 2 | [`2_conditional_workflow.ipynb`](./2_conditional_workflow.ipynb) | Conditional Workflow | Customer review router that classifies sentiment, diagnoses issue type/tone/urgency, and branches to a positive or negative response pipeline |
| 3 | [`3_iterative_workflow.ipynb`](./3_iterative_workflow.ipynb) | Iterative Workflow | Tweet generator with a self-evaluation loop — drafts a tweet, critiques it, rewrites it, and repeats until approved or max iterations reached |
| 4 | [`4_chatbot.ipynb`](./4_chatbot.ipynb) | Stateful Chatbot | Conversational agent with persistent message history using `InMemorySaver`, conditional exit detection, and proper `add_messages` reducer |
| 5 | *(coming soon)* | Tool Use & Agents | ReAct-style agents with external tool calling |
| 6 | *(coming soon)* | Human-in-the-Loop | Interrupt patterns, breakpoints, and manual approval steps |
| 7 | *(coming soon)* | Multi-Agent Systems | Supervisor and subgraph architectures |
| 8 | *(coming soon)* | Persistence & Memory | Long-term memory with checkpointers and stores |

---

## Repository Structure

```
langGraph_tutorial/
├── LangGraph_101.ipynb          # Introduction
├── 0_sequential_chains.ipynb    # Sequential workflow
├── 1_parallel_workflow.ipynb    # Parallel workflow
├── 2_conditional_workflow.ipynb # Conditional routing
├── 3_iterative_workflow.ipynb   # Iterative / self-correction loop
├── 4_chatbot.ipynb              # Stateful chatbot
└── .env                         # API keys (git-ignored)
```

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/vermaankit2005/langGraph-learning.git
cd langGraph-learning
```

**2. Install dependencies**
```bash
pip install langgraph langchain langchain-groq python-dotenv pydantic
```

**3. Configure environment**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

**4. Open any notebook and run**
```bash
jupyter notebook
```

---

## Learning Philosophy

This repository follows a deliberate progression — each notebook introduces exactly one new graph concept while building on everything that came before. The goal is not to collect examples, but to develop genuine intuition for how LangGraph models computation as a directed graph of stateful transformations.

---

## Author

**Ankit Verma** — [vermaankit.2005@gmail.com](mailto:vermaankit.2005@gmail.com)
