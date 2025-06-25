from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage
from typing import List, Dict
from dataclasses import dataclass

SYSTEM_PROMPT = """You are an AI assistant specializing in analyzing GitHub repositories. You have access to the contents of a repository, including code and text files, retrieved from a vector database.

Your role is to:
- Provide accurate, detailed explanations of the repository's content, including code functionality and documentation.
- Answer user queries based on the provided repository context.
- Explain technical concepts in clear, understandable language.
- Cite specific files from the repository when relevant.
- If the context is insufficient, state so and provide a general answer if possible.

Always base your responses on the provided repository context and be precise about which files support your claims."""

SYSTEM_TEMPLATE = """
---

## Context Snapshot:

- **Repository Files** (from vector DB):
{context}

- **Chat History**:
{chat_history}
"""

HUMAN_TEMPLATE = "**User Query**: {query}"

def format_chat_history(messages: List[BaseMessage]) -> str:
    if not messages:
        return ""
    formatted = []
    for msg in messages:
        role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)

def create_prompt_templates():
    conversation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", HUMAN_TEMPLATE),
        ]
    )
    return conversation_prompt

@dataclass
class PromptVariables:
    context: str
    chat_history: List[BaseMessage]
    query: str
    original_query: str | None = None

    @property
    def question(self) -> str:
        return self.original_query or self.query

    def to_dict(self) -> Dict:
        return {
            "context": self.context,
            "chat_history": format_chat_history(self.chat_history),
            "query": self.query,
            "question": self.question,
        }

def get_prompt_variables(
    context: str,
    chat_history: List[BaseMessage],
    query: str,
    original_query: str = None,
) -> Dict:
    vars = PromptVariables(
        context=context,
        chat_history=chat_history,
        query=query,
        original_query=original_query,
    )
    return vars.to_dict()