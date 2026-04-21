# 🔬 Research Assistant

A **source-aware multi-agent research pipeline** built with Streamlit, LangChain, and Mistral AI.

This app automates the full research workflow, from discovering sources to generating and evaluating structured reports.

---

## 🚀 Features

* 🔎 **Search Agent** — finds relevant and recent sources
* 📄 **Reader Agent** — extracts detailed content
* ✍️ **Writer Agent** — generates structured reports
* 🔍 **Evaluator Agent** — evaluates and scores output

* ⬇️ Download reports as Markdown

---

## 🧠 Architecture

Multi-agent pipeline:

```
Search → Reader → Writer → Evaluator
```

Each agent focuses on a specific task, improving modularity and output quality.

---

## 🛠️ Tech Stack

* Streamlit (UI)
* LangChain (agent orchestration)
* Mistral AI (LLM)
* Tavily (search API)

---

## ⚙️ Setup

### 1. Clone the repository

```
git clone https://github.com/RishabDey/ResearchAssistant.git
cd ResearchAssistant
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Add environment variables

Create a `.env` file:

```
OPENAI_API_KEY="your_key"
MISTRAL_API_KEY="your_key"
TAVILY_API_KEY="your_key"
GROQ_API_KEY="your_key"
GOOGLE_API_KEY="your_key"
```

---

### 4. Run the app

```
streamlit run app.py
```

---

## 🔒 Notes

* `.env` is ignored via `.gitignore`
* API keys are required for full functionality

---

## 📌 Future Improvements
* Memory-enabled agents
* Multi-source synthesis
* Citations & references
* Export to PDF

---

## 👤 Author
Rishab Dey

---

