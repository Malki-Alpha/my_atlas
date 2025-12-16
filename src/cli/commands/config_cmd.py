"""Config command to show and update configuration."""

import click
from ...utils.config import config
from ...utils.logger import get_logger

logger = get_logger(__name__)


@click.command(name='config')
@click.option(
    '--show',
    is_flag=True,
    help='Show current configuration'
)
@click.option(
    '--set',
    'set_value',
    type=str,
    help='Set a configuration value (key=value)'
)
def config(show: bool, set_value: str):
    """Show or update configuration settings.

    Examples:
        my-atlas config --show
        my-atlas config --set CHUNK_SIZE=512
    """
    click.echo("=" * 60)
    click.echo("MY ATLAS - Configuration")
    click.echo("=" * 60)

    if show or (not show and not set_value):
        # Show all configuration
        click.echo("\n[Embedding Configuration]")
        click.echo(f"  Model: {config.embedding.model}")
        click.echo(f"  Batch size: {config.embedding.batch_size}")
        click.echo(f"  Dimensions: {config.embedding.dimensions}")
        click.echo(f"  API key set: {'Yes' if config.embedding.api_key else 'No'}")

        click.echo("\n[Milvus Configuration]")
        click.echo(f"  Host: {config.milvus.host}")
        click.echo(f"  Port: {config.milvus.port}")
        click.echo(f"  Collection: {config.milvus.collection}")
        click.echo(f"  Index type: {config.milvus.index_type}")
        click.echo(f"  Metric type: {config.milvus.metric_type}")

        click.echo("\n[Chunking Configuration]")
        click.echo(f"  Chunk size: {config.chunking.chunk_size} tokens")
        click.echo(f"  Chunk overlap: {config.chunking.chunk_overlap} tokens")
        click.echo(f"  Method: {config.chunking.method}")

        click.echo("\n[Retrieval Configuration]")
        click.echo(f"  Hybrid top-k: {config.retrieval.hybrid_top_k}")
        click.echo(f"  Rerank top-k: {config.retrieval.rerank_top_k}")
        click.echo(f"  RRF k: {config.retrieval.rrf_k}")
        click.echo(f"  RRF alpha: {config.retrieval.rrf_alpha}")

        click.echo("\n[Reranker Configuration]")
        click.echo(f"  Model: {config.reranker.model}")
        click.echo(f"  Device: {config.reranker.device}")
        click.echo(f"  Max length: {config.reranker.max_length}")

        click.echo("\n[LLM Configuration]")
        click.echo(f"  Model: {config.llm.model}")
        click.echo(f"  Temperature: {config.llm.temperature}")
        click.echo(f"  Max tokens: {config.llm.max_tokens}")
        click.echo(f"  API key set: {'Yes' if config.llm.api_key else 'No'}")

        click.echo("\n[OCR Configuration]")
        click.echo(f"  Language: {config.ocr.lang}")
        click.echo(f"  Use GPU: {config.ocr.use_gpu}")
        click.echo(f"  Confidence threshold: {config.ocr.confidence_threshold}")

        click.echo("\n[Paths]")
        click.echo(f"  Data directory: {config.paths.data_dir}")
        click.echo(f"  Raw text: {config.paths.raw_text_dir}")
        click.echo(f"  Cleaned chunks: {config.paths.cleaned_chunks_dir}")
        click.echo(f"  Embedded chunks: {config.paths.emb_chunks_dir}")
        click.echo(f"  Logs: {config.paths.logs_dir}")

        # Validate API keys
        missing_keys = config.validate_api_keys()
        if missing_keys:
            click.echo("\n⚠️  WARNING: Missing API keys:")
            for key in missing_keys:
                click.echo(f"  • {key}")
            click.echo("\nPlease set these in your .env file.")

    elif set_value:
        click.echo("\n✗ Configuration updates via CLI not yet implemented.", err=True)
        click.echo("Please edit the .env file directly and restart the application.")

    click.echo("\n✓ Configuration displayed!")
