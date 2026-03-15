from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel, Field

# 1. We define the schemas for what our Agents will extract
class IssueOutput(BaseModel):
    core_issues: List[str] = Field(description="The main legal questions being decided")
    source_chunk_ids: List[str] = Field(description="IDs of the chunks where this was found")

class ArgumentsOutput(BaseModel):
    petitioner_args: List[str] = Field(description="Arguments made by the petitioner")
    respondent_args: List[str] = Field(description="Arguments made by the respondent")
    source_chunk_ids: List[str] = Field(description="IDs of the chunks where these were found")

class RuleOutput(BaseModel):
    statutes_applied: List[str] = Field(description="IPC/CrPC/BNS sections applied")
    precedents_cited: List[str] = Field(description="Past cases cited by the judge")
    source_chunk_ids: List[str] = Field(description="IDs of the chunks where these were found")

class ConclusionOutput(BaseModel):
    outcome: str = Field(description="The final verdict (e.g., Acquittal, Conviction upheld)")
    ratio_decidendi: str = Field(description="The core legal reasoning for the verdict")
    source_chunk_ids: List[str] = Field(description="IDs of the chunks where these were found")

# 2. We define the LangGraph State (The payload passed between agents)
class GraphState(TypedDict, total=False):
    case_id: str
    raw_text: str
    
    # This is where Agent 0 puts the color-coded paragraphs
    labeled_chunks: Dict[str, List[Dict[str, str]]]
    
    # This is where Agents 1a-1d put their extracted data
    extracted_issues: IssueOutput
    extracted_arguments: ArgumentsOutput
    extracted_rules: RuleOutput
    extracted_conclusion: ConclusionOutput
    
    # This is where Agent 2 puts the math scores (Updated to match engine.py)
    verification_score: float
    verification_status: str
    
    # This is the final output from Agent 3
    final_headnote: str