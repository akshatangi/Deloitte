import streamlit as st
import sqlite3
import ast 
import os

# Set up the web page styling
st.set_page_config(page_title="NYĀYA-INTELLIGENCE", page_icon="⚖️", layout="wide")

def load_data():
    """Fetches all processed cases from the SQLite database."""
    db_path = "nyaya_cases.db"
    if not os.path.exists(db_path):
        return []
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM cases")
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        
        # Convert database rows into a list of dictionaries
        cases = [dict(zip(column_names, row)) for row in rows]
    except sqlite3.OperationalError:
        cases = []
        
    conn.close()
    return cases

# --- DASHBOARD UI ---
st.title("⚖️ NYĀYA-INTELLIGENCE: AI Legal Compiler")
st.markdown("Automated intake, multi-agent extraction, and hallucination verification pipeline.")

cases = load_data()

if not cases:
    st.warning("⚠️ No cases found in the database. Please run `python engine.py` first to process a PDF!")
else:
    # 1. SIDEBAR: Case Navigation
    st.sidebar.header("📂 Processed Cases")
    # Clean up the file paths so the sidebar just shows the PDF name
    case_options = {case["case_id"].split(os.sep)[-1]: case for case in cases}
    
    selected_case_name = st.sidebar.selectbox("Select a Judgment to View:", list(case_options.keys()))
    selected_case = case_options[selected_case_name]
    
    st.divider()
    
    # 2. TOP SECTION: The Headnote and Quality Control Score
    col1, col2 = st.columns([1, 3])
    
    score = selected_case.get("verification_score", 0.0)
    
    with col1:
        st.markdown("### 🛡️ QC Score")
        # Dynamic colored badges based on the RapidFuzz math
        if score >= 80:
            st.success(f"**🟢 VERIFIED**\n\nLexical Match: {score}%")
        elif score >= 50:
            st.warning(f"**🟡 HUMAN REVIEW**\n\nLexical Match: {score}%")
        else:
            st.error(f"**🔴 FAILED QC**\n\nLexical Match: {score}%")
            
    with col2:
        st.markdown("### ✍️ SCC-Style AI Headnote")
        st.info(selected_case.get("headnote", "No headnote generated."))
        
    st.divider()
    
    # 3. BOTTOM SECTION: The Raw Extracted JSON Data
    st.markdown("### 🔍 Raw Extracted Data (Agents 1a-1d)")
    
    # We use tabs to keep the UI incredibly clean
    tab1, tab2, tab3 = st.tabs(["📌 Core Issues", "📖 Statutes & Rules", "⚖️ Final Outcome"])
    
    with tab1:
        issues_str = selected_case.get("issues", "[]")
        try:
            # Safely convert the stringified list back into a real Python list
            issue_list = ast.literal_eval(issues_str)
            for i, issue in enumerate(issue_list, 1):
                st.markdown(f"**{i}.** {issue}")
        except:
            st.write(issues_str)
            
    with tab2:
        statutes_str = selected_case.get("statutes", "[]")
        try:
            statute_list = ast.literal_eval(statutes_str)
            for s in statute_list:
                st.markdown(f"- {s}")
        except:
            st.write(statutes_str)
            
    with tab3:
        st.markdown(f"**Verdict Extracted:** {selected_case.get('outcome', 'Unknown')}")