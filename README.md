Phase 1: The Foundation (Agent 0 & State) - Setting up the environment, reading a PDF, and defining how data moves between agents.

Phase 2: The Paralegals (Agents 1a-1d) - Writing the prompts and Pydantic schemas to extract the IRAC data.

Phase 3: The Engine (LangGraph Orchestration & Verification) - Wiring the agents together and building the mathematical fact-checker (rapidfuzz).

Phase 4: The Brain (Knowledge Graph & RAG) - Pushing the final data into SQLite, ChromaDB, and NetworkX.