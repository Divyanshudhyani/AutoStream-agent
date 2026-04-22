# AutoStream AI Sales Agent

An AI-powered conversational sales assistant for **AutoStream**, a fictional SaaS video editing platform.

This project was built as part of the **Inflx Assignment** for **ServiceHive**, with the goal of simulating how an AI sales assistant can:

* understand customer intent,
* answer product/pricing questions,
* collect lead details conversationally,
* and prepare the workflow for channels like **WhatsApp**.

The system is powered by **LangGraph**, uses **Gemini Flash** for language understanding and response generation, and includes a lightweight **RAG pipeline** over a local knowledge base.

---

## What this agent does

The assistant can:

* greet users naturally,
* answer questions about pricing, plans, and product features,
* detect when a user shows buying intent,
* collect lead details (name, email, platform),
* trigger a lead capture function only when all required data is available.

Example flow:

```bash
User: Hi
Bot: Hey! Welcome to AutoStream. How can I help you today?

User: What plans do you offer?
Bot: We currently offer Basic and Pro plans...

User: I want to try the Pro plan
Bot: Great! Can I get your full name?

User: Divyanshu Dhyani
Bot: Thanks! What's your email address?

User: divyanshu@example.com
Bot: Perfect. Which platform do you create content for?

User: YouTube
Bot: Awesome — you're all set! Our team will contact you shortly.
```

---

## Tech Stack

* **Python 3.9+**
* **LangGraph**
* **LangChain**
* **Google Gemini Flash**
* **Local JSON Knowledge Base**
* **Regex + Heuristic Lead Extraction**

---

## Project Structure

```bash
autostream-agent/
│
├── main.py
├── requirements.txt
├── .env 
│
├── knowledge_base/
│   └── autostream_kb.json
│
├── agent/
│   ├── graph.py
│   ├── intent_detector.py
│   └── rag_pipeline.py
│
└── tools/
    └── lead_capture.py
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone 
cd autostream-agent
```

---

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add your Gemini API key

Create a `.env` file:

```bash
GEMINI_API_KEY=your_api_key_here
```

You can generate a free API key from **Google AI Studio**.

---

### 5. Run the assistant

```bash
python main.py
```

Optional:

```bash
python main.py --debug
python main.py --verbose
```

---

## Architecture Overview

This project uses a **stateful workflow** built with LangGraph.

Each user message goes through the following steps:

### 1. Intent Detection

A Gemini model classifies the latest message into one of three intents:

* `greeting`
* `inquiry`
* `high_intent`

This helps the assistant understand whether the user is casually chatting, asking questions, or is ready to sign up.

---

### 2. Context Retrieval

The system retrieves relevant pricing/product information from a local JSON knowledge base using a simple keyword-based RAG approach.

This avoids hallucinations and ensures responses remain grounded in business data.

---

### 3. Lead Extraction

When the user shows buying intent, the assistant begins collecting:

* name
* email
* content platform

Regex + heuristics are used to extract these fields incrementally.

---

### 4. Lead Capture

Once all required fields are available, a mock lead capture tool is triggered.

This ensures the lead is only captured when complete.

---

### 5. Response Generation

Gemini generates the assistant response using:

* detected intent,
* retrieved knowledge,
* lead collection state,
* recent chat history.

This keeps replies contextual and natural.

---

## Why LangGraph?

LangGraph was chosen because it gives explicit control over:

* state transitions,
* node execution,
* lead collection flow,
* tool invocation conditions.

Compared to multi-agent frameworks, it is lightweight and better suited for deterministic workflows like:

```bash
Intent → Retrieve Context → Extract Lead → Generate Response
```

---

## WhatsApp Deployment Plan

The current version runs as a CLI chatbot, but it can be deployed to **WhatsApp Business API** with a webhook-based architecture.

### Deployment flow:

1. WhatsApp sends incoming message to webhook
2. Backend loads conversation state
3. LangGraph agent processes message
4. Response is sent back to WhatsApp API
5. Updated state is stored in Redis

This allows the same conversational logic to work across multiple users while preserving session state.

Suggested deployment stack:

* **FastAPI**
* **Redis**
* **Meta WhatsApp Cloud API**
* **Railway / Render / AWS**

---

## Key Features Implemented

* Intent classification using Gemini
* Retrieval-augmented responses from local KB
* Stateful lead collection
* Conditional tool execution
* Multi-turn conversational memory
* WhatsApp-ready architecture

---

## Future Improvements

Given more time, the next improvements would be:

* switch from keyword retrieval to vector embeddings,
* integrate real CRM/webhook lead capture,
* add Redis-backed session persistence,
* deploy as an API service,
* add analytics for conversion tracking.

---

## Assignment Goals Covered

This project demonstrates:

* conversational AI workflow design,
* intent classification,
* retrieval augmentation,
* state management,
* lead qualification logic,
* deployment planning.

The focus was to build something **modular, production-oriented, and easy to extend**, while keeping the architecture simple enough to understand and maintain.
# AutoStream-agent
