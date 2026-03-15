import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load the secret API key from the .env file
load_dotenv()

# Initialize the Big Brain (Groq + Llama 3.3)
def get_llm():
    return ChatGroq(
        # Llama 3.3 70B is currently one of the smartest/fastest open models on Groq
        model="llama-3.3-70b-versatile", 
        temperature=0.1, # Keep it low so it doesn't hallucinate
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

# Test to make sure it's connected
if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("What is the full form of IPC in Indian law?")
    print("Groq API Connection Successful! Response:", response.content)