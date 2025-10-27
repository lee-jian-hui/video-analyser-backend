"""
Chat History Prompts

Centralized prompt templates for chat history summarization and merging.
"""

from typing import List


class ChatHistoryPrompts:
    """Prompt templates for summarizing and merging chat histories."""

    @staticmethod
    def format_summarize_messages_prompt(messages_text: str) -> str:
        """Build a prompt to summarize recent messages.

        The LLM should respond starting with:
        "Here is the summary of your past conversation: ..."
        """
        return f"""Summarize this conversation excerpt concisely (2-3 sentences):

{messages_text}

Focus on:
1. What the user asked about
2. Key findings or results
3. Any important context

Important formatting requirement:
- Begin your response with exactly: "Here is the summary of your past conversation: "

Summary:
"""

    @staticmethod
    def format_merge_summaries_prompt(old_summary: str, new_summary: str) -> str:
        """Build a prompt to merge two summaries into a concise one.

        Preserve the same opening format for consistency.
        """
        return f"""Merge these two conversation summaries into one concise summary (3-4 sentences max):

Previous summary:
{old_summary}

New summary:
{new_summary}

Important formatting requirement:
- Begin your response with exactly: "Here is the summary of your past conversation: "

Combined summary:
"""

