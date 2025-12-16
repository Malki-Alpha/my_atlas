"""LLM client for GPT-4o-mini inference."""

from typing import List, Dict, Any
from openai import OpenAI
from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based ONLY on the provided context.

Instructions:
1. Answer the question using ONLY information from the context below
2. If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question"
3. Always cite the source page number when providing information
4. Be concise and factual
5. Do not use external knowledge or make assumptions"""


FALLBACK_SYSTEM_PROMPT = """You are a helpful AI assistant with general knowledge.

IMPORTANT: The user's question could not be answered from their knowledge base.
Provide a general answer using your training data, but:
1. Clearly state this is general information, not from their documents
2. Note that the information may not be up-to-date
3. Keep the answer concise
4. Suggest they verify from authoritative sources"""


class LLMClient:
    """Client for GPT-4o-mini inference and answer generation."""

    def __init__(self):
        """Initialize OpenAI client."""
        self.config = config.llm

        if not self.config.api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")

        self.client = OpenAI(api_key=self.config.api_key)

        logger.info(f"LLMClient initialized with model: {self.config.model}")

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format context chunks for the prompt.

        Args:
            chunks: List of ranked chunks

        Returns:
            Formatted context string
        """
        context_parts = []

        for chunk in chunks:
            metadata = chunk['metadata']
            doc_name = metadata.get('document_name', 'Unknown')
            page_num = metadata.get('page_number', '?')

            context_parts.append(
                f"[Source: {doc_name}, Page {page_num}]\n{chunk['content']}"
            )

        return "\n\n".join(context_parts)

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate answer using GPT-4o-mini with context.

        Args:
            query: User question
            context_chunks: Top K reranked chunks

        Returns:
            dict with answer, sources, and metadata
        """
        logger.info(f"Generating answer for query: '{query[:50]}...'")

        # Format context
        context_text = self._format_context(context_chunks)

        user_message = f"""Context:
{context_text}

Question: {query}

Answer:"""

        # Call GPT-4o-mini
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )

            answer = response.choices[0].message.content

            # Extract sources
            sources = list(set([
                f"{chunk['metadata']['document_name']} (p. {chunk['metadata']['page_number']})"
                for chunk in context_chunks
            ]))

            result = {
                "answer": answer,
                "sources": sources,
                "model": self.config.model,
                "is_fallback": False,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            }

            logger.info(f"Answer generated ({result['tokens']['total']} tokens)")

            return result

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def generate_fallback_answer(self, query: str) -> Dict[str, Any]:
        """Generate generic answer when grounded answer fails validation.

        Args:
            query: User question

        Returns:
            dict with fallback answer and warning
        """
        logger.info(f"Generating fallback answer for: '{query[:50]}...'")

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ],
                temperature=self.config.fallback_temperature,
                max_tokens=self.config.fallback_max_tokens
            )

            answer = response.choices[0].message.content

            # Add warning prefix
            warning = "⚠️ No answer found in your knowledge base. Here's general information (may not be up-to-date):\n\n"

            result = {
                "answer": warning + answer,
                "sources": ["General AI knowledge (not from documents)"],
                "model": self.config.model,
                "is_fallback": True,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            }

            logger.info(f"Fallback answer generated ({result['tokens']['total']} tokens)")

            return result

        except Exception as e:
            logger.error(f"Error generating fallback answer: {e}")
            raise
