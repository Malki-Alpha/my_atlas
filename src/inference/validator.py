"""Answer validation module."""

from openai import OpenAI
from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


VALIDATION_PROMPT = """You are an answer quality validator.

Given a question and an answer, determine if the answer adequately addresses the question.

An answer is VALID if:
- It directly addresses the question asked
- It provides specific information (not just "I don't know")
- It's based on the provided context

An answer is INVALID if:
- It says there's not enough information
- It doesn't address the question
- It's too vague or generic

Respond with ONLY "VALID" or "INVALID" followed by a brief one-sentence explanation."""


class AnswerValidator:
    """Validate if answer adequately addresses the query."""

    def __init__(self):
        """Initialize OpenAI client for validation."""
        self.config = config.llm

        if not self.config.api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")

        self.client = OpenAI(api_key=self.config.api_key)

        logger.info("AnswerValidator initialized")

    def validate(self, query: str, answer: str) -> dict:
        """Validate if answer adequately addresses the query.

        Args:
            query: Original user question
            answer: Generated answer

        Returns:
            dict with is_valid bool and explanation
        """
        logger.debug(f"Validating answer for query: '{query[:50]}...'")

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": VALIDATION_PROMPT},
                    {"role": "user", "content": f"Question: {query}\n\nAnswer: {answer}"}
                ],
                temperature=0.0,
                max_tokens=100
            )

            validation_result = response.choices[0].message.content

            is_valid = validation_result.upper().startswith("VALID")

            logger.debug(f"Validation result: {'VALID' if is_valid else 'INVALID'}")

            return {
                "is_valid": is_valid,
                "explanation": validation_result
            }

        except Exception as e:
            logger.error(f"Error during validation: {e}")
            # Default to valid on error to avoid blocking
            return {
                "is_valid": True,
                "explanation": f"Validation error: {e}"
            }
