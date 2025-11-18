RAG_AGENT_SYSTEM_PROMPT = """You are a helpful assistant. Your task is to answer the user's question based on the provided context.
If the context is not relevant to the question, answer with your general knowledge but mention that the context is not relevant to the question.
If you do not have enough information to answer the question, just tell the user that you do not have enough information to answer the question.
"""

RAG_USER_PROMPT = """Based on the context provided, answer the user's question. 

Question: 
{question}

Context: 
{context}

Answer: """