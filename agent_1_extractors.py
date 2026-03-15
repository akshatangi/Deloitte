from langchain_core.prompts import ChatPromptTemplate
from llm_setup import get_llm
from state import IssueOutput, ArgumentsOutput, RuleOutput, ConclusionOutput

# --- AGENT 1a: THE ISSUE EXTRACTOR ---
def run_agent_1a_issue(fact_chunks: list, reasoning_chunks: list) -> IssueOutput:
    print("🔍 [Agent 1a] Extracting Legal Issues...")
    llm = get_llm().with_structured_output(IssueOutput)
    
    context_text = ""
    for chunk in fact_chunks[:10] + reasoning_chunks[:5]: 
        context_text += f"[ID: {chunk['id']}] {chunk['text']}\n\n"
        
    if not context_text.strip():
        return IssueOutput(core_issues=["No relevant facts provided"], source_chunk_ids=[])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Indian Supreme Court lawyer. 
        Extract 1 to 3 core 'Issues' (the main legal questions being decided).
        Also, return the [ID: chunk_xyz] of the exact chunks where you found this information."""),
        ("human", "Here is the text:\n\n{context}")
    ])
    
    return (prompt | llm).invoke({"context": context_text})

# --- AGENT 1b: THE ARGUMENTS EXTRACTOR ---
def run_agent_1b_arguments(pet_chunks: list, resp_chunks: list) -> ArgumentsOutput:
    print("🗣️ [Agent 1b] Extracting Arguments...")
    llm = get_llm().with_structured_output(ArgumentsOutput)
    
    context_text = ""
    for chunk in pet_chunks[:10]: 
        context_text += f"[PETITIONER CHUNK ID: {chunk['id']}] {chunk['text']}\n\n"
    for chunk in resp_chunks[:10]: 
        context_text += f"[RESPONDENT CHUNK ID: {chunk['id']}] {chunk['text']}\n\n"
        
    if not context_text.strip():
        return ArgumentsOutput(petitioner_args=[], respondent_args=[], source_chunk_ids=[])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Indian appellate lawyer. 
        Extract the main arguments made by the Petitioner and the main arguments made by the Respondent.
        Keep them concise and professional.
        Also, return the [ID: chunk_xyz] of the exact chunks where you found this information."""),
        ("human", "Here are the arguments:\n\n{context}")
    ])
    
    return (prompt | llm).invoke({"context": context_text})

# --- AGENT 1c: THE RULE EXTRACTOR ---
def run_agent_1c_rule(statute_chunks: list, precedent_chunks: list) -> RuleOutput:
    print("📖 [Agent 1c] Extracting Statutes and Precedents...")
    llm = get_llm().with_structured_output(RuleOutput)
    
    context_text = ""
    for chunk in statute_chunks[:10] + precedent_chunks[:10]: 
        context_text += f"[ID: {chunk['id']}] {chunk['text']}\n\n"
        
    if not context_text.strip():
        return RuleOutput(statutes_applied=[], precedents_cited=[], source_chunk_ids=[])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Indian legal researcher. 
        Extract the specific Statutes/Sections applied (e.g., Section 302 IPC, NDPS Act) 
        and the specific Precedents/Case Laws cited by the court.
        Also, return the [ID: chunk_xyz] of the exact chunks where you found this information."""),
        ("human", "Here is the text:\n\n{context}")
    ])
    
    return (prompt | llm).invoke({"context": context_text})

# --- AGENT 1d: THE CONCLUSION EXTRACTOR ---
def run_agent_1d_conclusion(reasoning_chunks: list, order_chunks: list) -> ConclusionOutput:
    print("🔨 [Agent 1d] Extracting Final Verdict...")
    llm = get_llm().with_structured_output(ConclusionOutput)
    
    context_text = ""
    for chunk in reasoning_chunks[-5:] + order_chunks[-10:]: 
        context_text += f"[ID: {chunk['id']}] {chunk['text']}\n\n"
        
    if not context_text.strip():
        return ConclusionOutput(outcome="Unknown", ratio_decidendi="Unknown", source_chunk_ids=[])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Indian Supreme Court judge. 
        Extract the final 'Outcome' (e.g., Appeal Dismissed, Conviction Upheld) 
        and the 'Ratio Decidendi' (a 1-2 sentence summary of the core legal reasoning for the verdict).
        Also, return the [ID: chunk_xyz] of the exact chunks where you found this information."""),
        ("human", "Here is the text:\n\n{context}")
    ])
    
    return (prompt | llm).invoke({"context": context_text})

# --- TEST ALL 4 PARALEGALS ---
if __name__ == "__main__":
    print("--- BOOTING UP THE AI PARALEGAL TEAM ---")
    
    # Dummy data to test the new agents
    dummy_pet = [{"id": "chunk_pet1", "text": "The appellant argued the confession was coerced."}]
    dummy_resp = [{"id": "chunk_res1", "text": "The state argued the confession was voluntary and corroborated by forensics."}]
    dummy_stat = [{"id": "chunk_stat1", "text": "The court examined Section 27 of the Indian Evidence Act."}]
    dummy_prec = [{"id": "chunk_prec1", "text": "Relying on State of UP v. Deoman Upadhyaya, the court noted..."}]
    dummy_reas = [{"id": "chunk_reas1", "text": "The forensic evidence heavily corroborates the voluntary nature of the statement."}]
    dummy_ord = [{"id": "chunk_ord1", "text": "The appeal is dismissed. The conviction under Section 302 IPC is upheld."}]
    
    # Run them!
    args_out = run_agent_1b_arguments(dummy_pet, dummy_resp)
    rule_out = run_agent_1c_rule(dummy_stat, dummy_prec)
    conc_out = run_agent_1d_conclusion(dummy_reas, dummy_ord)
    
    print("\n✅ --- EXTRACTION COMPLETE ---")
    print(f"Petitioner Args: {args_out.petitioner_args}")
    print(f"Statutes Found: {rule_out.statutes_applied}")
    print(f"Precedents Found: {rule_out.precedents_cited}")
    print(f"Final Outcome: {conc_out.outcome}")