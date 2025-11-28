# Retail Analytics Copilot (DSPy + LangGraph)

This project implements a local, offline AI agent for answering retail analytics questions using the Northwind database. It combines RAG (Retrieval-Augmented Generation) for policy/calendar questions and NL2SQL (Natural Language to SQL) for data queries, orchestrated by a LangGraph state machine.

## Graph Design
The agent uses a **7-node LangGraph architecture** with a stateful repair loop:
*   **Router**: Classifies queries into `rag`, `sql`, or `hybrid` (combining docs + data) using DSPy.
*   **Retriever & Planner**: Fetches documentation (policies, marketing calendar) and extracts constraints (dates, KPIs).
*   **NL2SQL & Executor**: Generates SQLite queries via a DSPy module and executes them against the local Northwind database.
*   **Synthesizer**: Combines SQL results and retrieved context to produce a strictly typed answer with citations.
*   **Repair Loop**: Automatically retries up to 2 times if SQL execution fails or the output format is incorrect.

## DSPy Module Optimization
I attempted to optimize the **NL2SQL Module** using three different strategies to improve SQL generation accuracy.

**Metric Delta (Valid SQL & Execution Success):**
*   **Method 1: BootstrapFewShot**
    *   **Before:** 70.0%
    *   **After:** 70.0%
    *   **Result:** No improvement (+0.0%). The base model performance was already strong for the test set, or the examples did not provide additional generalization power for this specific model/dataset pair.


*   **Method 3: LabeledFewShot**
    *   **Result:** Failed. This method caused the model to hallucinate, producing invalid SQL or hallucinations in the generated queries.

## Trade-offs & Assumptions
*   **CostOfGoods Approximation**: The Northwind database lacks a direct "Cost" field for products. As per the assignment instructions, I assume `CostOfGoods ≈ 0.7 * UnitPrice` for calculating Gross Margin.
*   **Local Model Constraints**: Built for `phi3.5:3.8b-mini-instruct`. Prompts are kept compact to fit within the smaller context window and ensure reliable instruction following on a quantized local model.
*   **Strict Repair Limits**: The repair loop is hard-coded to a maximum of 2 retries to prevent potential infinite loops and reduce inference latency, trading off some resilience for speed.
