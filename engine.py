import os
from langgraph.graph import StateGraph, START, END
from state import GraphState

# Import our custom agents
from agent_0_ingestion import extract_text_from_pdf, semantic_chunker, opennyai_role_classifier
from agent_1_extractors import (
    run_agent_1a_issue, run_agent_1b_arguments, 
    run_agent_1c_rule, run_agent_1d_conclusion
)
from agent_2_critic import verify_extraction
from agent_3_compiler import setup_database, generate_headnote, save_to_database

print("⚙️ Initializing NYĀYA-INTELLIGENCE LangGraph Engine...")

# --- NODE DEFINITIONS ---
def node_intake_desk(state: GraphState):
    print("\n--- [NODE: Intake Desk] ---")
    raw_text = extract_text_from_pdf(state["case_id"]) 
    chunks = semantic_chunker(raw_text)
    tagged_data = opennyai_role_classifier(chunks)
    return {"raw_text": raw_text, "labeled_chunks": tagged_data}

def node_extract_issue(state: GraphState):
    print("\n--- [NODE: Paralegal 1a (Issue)] ---")
    chunks = state.get("labeled_chunks", {})
    result = run_agent_1a_issue(chunks.get("FACT", []), chunks.get("REASONING", []))
    return {"extracted_issues": result}

def node_extract_arguments(state: GraphState):
    print("\n--- [NODE: Paralegal 1b (Arguments)] ---")
    chunks = state.get("labeled_chunks", {})
    result = run_agent_1b_arguments(chunks.get("PETITIONER", []), chunks.get("RESPONDENT", []))
    return {"extracted_arguments": result}

def node_extract_rule(state: GraphState):
    print("\n--- [NODE: Paralegal 1c (Rule)] ---")
    chunks = state.get("labeled_chunks", {})
    result = run_agent_1c_rule(chunks.get("STATUTE", []), chunks.get("PRECEDENT", []))
    return {"extracted_rules": result}

def node_extract_conclusion(state: GraphState):
    print("\n--- [NODE: Paralegal 1d (Conclusion)] ---")
    chunks = state.get("labeled_chunks", {})
    result = run_agent_1d_conclusion(chunks.get("REASONING", []), chunks.get("ORDER", []))
    return {"extracted_conclusion": result}

def node_quality_control(state: GraphState):
    print("\n--- [NODE: Quality Control (Critic)] ---")
    tagged_chunks = state.get("labeled_chunks", {})
    conclusion_data = state.get("extracted_conclusion")
    
    if conclusion_data and hasattr(conclusion_data, 'outcome'):
        verification_report = verify_extraction(
            extracted_items=[conclusion_data.outcome, conclusion_data.ratio_decidendi],
            source_chunk_ids=conclusion_data.source_chunk_ids,
            all_tagged_chunks=tagged_chunks
        )
        return {"verification_score": verification_report["score"], "verification_status": verification_report["status"]}
    else:
        return {"verification_score": 0.0, "verification_status": "🔴 FAILED (No conclusion data)"}

def node_master_compiler(state: GraphState):
    print("\n--- [NODE: Master Compiler] ---")
    final_headnote = generate_headnote(state)
    save_to_database(state["case_id"], state, final_headnote)
    return {"final_headnote": final_headnote}

# --- BUILD THE GRAPH ---
builder = StateGraph(GraphState)

builder.add_node("Intake", node_intake_desk)
builder.add_node("Paralegal_Issue", node_extract_issue)
builder.add_node("Paralegal_Args", node_extract_arguments)
builder.add_node("Paralegal_Rule", node_extract_rule)
builder.add_node("Paralegal_Conclusion", node_extract_conclusion)
builder.add_node("Quality_Control", node_quality_control) 
builder.add_node("Master_Compiler", node_master_compiler)

builder.add_edge(START, "Intake")

# Parallel Execution
builder.add_edge("Intake", "Paralegal_Issue")
builder.add_edge("Intake", "Paralegal_Args")
builder.add_edge("Intake", "Paralegal_Rule")
builder.add_edge("Intake", "Paralegal_Conclusion")

# All paralegals dump their work into Quality Control
builder.add_edge("Paralegal_Issue", "Quality_Control")
builder.add_edge("Paralegal_Args", "Quality_Control")
builder.add_edge("Paralegal_Rule", "Quality_Control")
builder.add_edge("Paralegal_Conclusion", "Quality_Control")

builder.add_edge("Quality_Control", "Master_Compiler") 
builder.add_edge("Master_Compiler", END) 

graph = builder.compile()

# --- THE ENTERPRISE BATCH PROCESSOR ---
if __name__ == "__main__":
    setup_database() # Make sure the database exists
    
    # Point to your folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(script_dir, "input_pdfs")
    
    # Grab every PDF in the folder
    if not os.path.exists(pdf_folder):
        print(f"❌ Folder not found at: {pdf_folder}")
    else:
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"❌ No PDFs found in: {pdf_folder}")
        else:
            print(f"\n📂 Found {len(pdf_files)} PDFs. Starting Enterprise Batch Processing...\n")
            
            # Loop through them all!
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_folder, pdf_file)
                
                print(f"==================================================")
                print(f"🚀 NOW PROCESSING: {pdf_file}")
                print(f"==================================================")
                
                # Run the pipeline for this specific PDF
                initial_state = {"case_id": pdf_path}
                final_state = graph.invoke(initial_state)
                
                print(f"\n✅ Successfully finished and saved: {pdf_file}\n")
                
            print("🎉 BATCH JOB COMPLETE! ALL 10 CASES SAVED TO DATABASE!")