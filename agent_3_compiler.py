import sqlite3
from langchain_core.prompts import ChatPromptTemplate
from llm_setup import get_llm

def setup_database():
    """Creates a local SQLite database to store our final case summaries."""
    conn = sqlite3.connect("nyaya_cases.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            issues TEXT,
            statutes TEXT,
            outcome TEXT,
            headnote TEXT,
            verification_score REAL
        )
    ''')
    conn.commit()
    conn.close()
    print("🗄️ Database 'nyaya_cases.db' is ready.")

def generate_headnote(state_data: dict) -> str:
    """Agent 3: Takes all extracted JSON data and writes a flowing, professional legal summary."""
    print("✍️ [Agent 3] Drafting final SCC-style Headnote...")
    
    # We use a standard text-based LLM here, no strict JSON needed for the draft!
    llm = get_llm() 
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Senior Editor of Supreme Court Cases (SCC). Write a professional, 3-paragraph legal headnote summarizing this case based ONLY on the provided verified data."),
        ("human", """
        Draft a headnote using this data:
        ISSUES: {issues}
        ARGUMENTS: {arguments}
        STATUTES: {statutes}
        OUTCOME & REASONING: {conclusion}
        """)
    ])
    
    chain = prompt | llm
    
    result = chain.invoke({
        "issues": state_data.get("extracted_issues"),
        "arguments": state_data.get("extracted_arguments"),
        "statutes": state_data.get("extracted_rules"),
        "conclusion": state_data.get("extracted_conclusion")
    })
    
    return result.content

def save_to_database(case_id: str, state_data: dict, headnote: str):
    """Saves the final compiled data into SQLite."""
    print(f"💾 [Agent 3] Saving case to secure database...")
    
    # Safely extract data, falling back to empty strings if something is missing
    issues = str(getattr(state_data.get('extracted_issues'), 'core_issues', []))
    statutes = str(getattr(state_data.get('extracted_rules'), 'statutes_applied', []))
    outcome = getattr(state_data.get('extracted_conclusion'), 'outcome', "Unknown")
    score = state_data.get('verification_score', 0.0)
    
    conn = sqlite3.connect("nyaya_cases.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO cases (case_id, issues, statutes, outcome, headnote, verification_score)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (case_id, issues, statutes, outcome, headnote, score))
    
    conn.commit()
    conn.close()
    print("✅ Successfully written to nyaya_cases.db!")