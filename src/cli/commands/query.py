"""Query command for chatbot interactions."""

import click
from ...retrieval.hybrid_retriever import HybridRetriever
from ...retrieval.reranker import JinaReranker
from ...inference.llm_client import LLMClient
from ...inference.validator import AnswerValidator
from ...utils.logger import get_logger
from ...utils.config import config

logger = get_logger(__name__)


@click.command()
@click.argument('question')
@click.option(
    '--top-k',
    type=int,
    default=None,
    help=f'Number of chunks to retrieve (default: {config.retrieval.rerank_top_k})'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show detailed retrieval information'
)
@click.option(
    '--no-fallback',
    is_flag=True,
    help='Disable fallback to generic answers'
)
def query(question: str, top_k: int, verbose: bool, no_fallback: bool):
    """Query the knowledge base with a question.

    QUESTION: Your question (use quotes for multi-word questions)

    Examples:
        my-atlas query "What is RAG?"
        my-atlas query "How does hybrid search work?" --verbose
        my-atlas query "Explain embeddings" --top-k 10
    """
    click.echo("=" * 60)
    click.echo("MY ATLAS - Query Knowledge Base")
    click.echo("=" * 60)
    click.echo(f"\nQuestion: {question}\n")

    try:
        # Initialize components
        if verbose:
            click.echo("Initializing retrieval components...")

        retriever = HybridRetriever()
        reranker = JinaReranker()
        llm_client = LLMClient()
        validator = AnswerValidator()

        # Step 1: Hybrid retrieval
        if verbose:
            click.echo(f"\n[1/4] Performing hybrid search (semantic + BM25)...")

        hybrid_results = retriever.search(question, top_k=config.retrieval.hybrid_top_k)

        if verbose:
            click.echo(f"  Retrieved {len(hybrid_results)} candidates")

        if not hybrid_results:
            click.echo("\n✗ No relevant documents found in the knowledge base.", err=True)
            return

        # Step 2: Reranking
        if verbose:
            click.echo(f"\n[2/4] Reranking with Jina Reranker v2...")

        if top_k is None:
            top_k = config.retrieval.rerank_top_k

        reranked_results = reranker.rerank(question, hybrid_results, top_k=top_k)

        if verbose:
            click.echo(f"  Top {len(reranked_results)} chunks after reranking")
            click.echo("\n  Top results:")
            for i, result in enumerate(reranked_results[:3], 1):
                doc_name = result['metadata']['document_name']
                page = result['metadata']['page_number']
                score = result['rerank_score']
                preview = result['content'][:100].replace('\n', ' ')
                click.echo(f"    {i}. [{doc_name}, p.{page}] (score: {score:.4f})")
                click.echo(f"       \"{preview}...\"")

        # Step 3: Generate answer
        if verbose:
            click.echo(f"\n[3/4] Generating answer with GPT-4o-mini...")

        answer_result = llm_client.generate_answer(question, reranked_results)

        if verbose:
            click.echo(f"  Tokens used: {answer_result['tokens']['total']}")

        # Step 4: Validate answer
        if verbose:
            click.echo(f"\n[4/4] Validating answer...")

        validation = validator.validate(question, answer_result['answer'])

        if verbose:
            click.echo(f"  Validation: {'VALID' if validation['is_valid'] else 'INVALID'}")
            click.echo(f"  Reason: {validation['explanation']}")

        # Use fallback if invalid and fallback not disabled
        if not validation['is_valid'] and not no_fallback:
            if verbose:
                click.echo("\n  Using fallback (general knowledge)...")

            answer_result = llm_client.generate_fallback_answer(question)

        # Display answer
        click.echo("\n" + "=" * 60)
        click.echo("ANSWER")
        click.echo("=" * 60)
        click.echo(f"\n{answer_result['answer']}\n")

        # Display sources
        click.echo("=" * 60)
        click.echo("SOURCES")
        click.echo("=" * 60)
        for source in answer_result['sources']:
            click.echo(f"  • {source}")

        # Display metadata
        if verbose or answer_result.get('is_fallback', False):
            click.echo("\n" + "=" * 60)
            click.echo("METADATA")
            click.echo("=" * 60)
            click.echo(f"  Model: {answer_result['model']}")
            click.echo(f"  Fallback: {answer_result.get('is_fallback', False)}")
            click.echo(f"  Tokens: {answer_result['tokens']['total']}")
            click.echo(f"  Validation: {'VALID' if validation['is_valid'] else 'INVALID'}")

        click.echo("\n✓ Query complete!")

    except Exception as e:
        click.echo(f"\n✗ Error during query: {e}", err=True)
        logger.error(f"Query error: {e}", exc_info=True)
        raise click.Abort()
