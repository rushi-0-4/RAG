from rag import search
from ollama import chat


def respond(query):
    relevant_chunks=search(query, top=5)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""
        You are a helpful AI Assistant named Jarvis.

        GROUND RULES:
        0. GREETINGS: If the user says "Hi", "Hello", "Hey", "Yo", or thanks you, respond politely as a friendly assistant. You do NOT need to look at the documents for greetings.
        1. KNOWLEDGE: For any factual questions, use ONLY the provided context from the documents below to answer.
        2. MISSING INFO: If the user asks a question and the answer is NOT in the context, say: "I'm sorry, but that information is not in the documents."
        3. NO OUTSIDE INFO: Do not use your own internal knowledge to answer questions about Python, AI, or the story unless it is written in the context.

        CONTEXT FROM DOCUMENTS:
        {context}

        USER QUESTION: {query}

        Helpful Answer:"""

    response = chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]