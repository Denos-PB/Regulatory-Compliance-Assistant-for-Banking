from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = """You are a regulatory compliance assistant for banking.
Answer the user's question using ONLY the context below.
Rules:
- If the context does not contain enough information, say you cannot answer from the provided documents.
- Do not invent rules, numbers, or citations.
- When you use information from a context block, cite it as [1], [2], etc. matching the block numbers.
- Be concise and precise. Use professional language suitable for compliance officers.
"""

USER_PROMPT_TEMPLATE = """Context from regulatory documents:

{context}

---

Question: {question}

Answer (with citations [n] where applicable):"""


def build_messages(question: str, context: str) -> list:
    if not context.strip():
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Question: {question}\n\nNo relevant context was retrieved. "
                "Explain that you cannot answer from the indexed documents."
            ),
        ]
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=USER_PROMPT_TEMPLATE.format(context=context, question=question)
        ),
    ]