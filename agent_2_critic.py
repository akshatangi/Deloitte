from rapidfuzz import fuzz

def verify_extraction(extracted_items: list, source_chunk_ids: list, all_tagged_chunks: dict) -> dict:
    """
    The Critic-Verifier: Calculates a confidence score by fuzzy matching 
    the LLM's output against the exact paragraphs it claims to have read.
    """
    print("🛡️ [Agent 2] Critic-Verifier scanning for hallucinations...")
    
    # 1. Flatten our chunk dictionary so we can easily find chunks by their ID
    chunk_map = {}
    for role, chunks in all_tagged_chunks.items():
        for chunk in chunks:
            chunk_map[chunk['id']] = chunk['text']
            
    # 2. Rebuild the exact text the LLM read
    source_text = ""
    for c_id in source_chunk_ids:
        if c_id in chunk_map:
            source_text += chunk_map[c_id] + " "
            
    if not source_text.strip():
        return {"score": 0.0, "status": "🔴 FAILED (No Source Found)", "requires_human": True}

    # 3. Mathematically check the LLM's output against the source text
    # We use 'token_set_ratio' which ignores word order and focuses on core keyword intersection
    total_score = 0
    valid_items = [item for item in extracted_items if item and str(item).strip() != ""]
    
    if not valid_items:
        return {"score": 0.0, "status": "🔴 FAILED (Empty Output)", "requires_human": True}
        
    for item in valid_items:
        score = fuzz.token_set_ratio(str(item).lower(), source_text.lower())
        total_score += score
        
    avg_score = total_score / len(valid_items)
    
    # 4. Apply our Hackathon Threshold (e.g., if it drops below 65%, flag it!)
    threshold = 65.0
    requires_human = avg_score < threshold
    
    if requires_human:
        status = "🟡 WARNING: Low Confidence. Flagged for Human Review."
    else:
        status = "🟢 VERIFIED: Safe to proceed."
        
    return {
        "score": round(avg_score, 2),
        "status": status,
        "requires_human": requires_human
    }