"""Config command to show and update configuration."""

import click
from ...utils.config import config as app_config
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
        click.echo(f"  Model: {app_config.embedding.model}")
        click.echo(f"  Batch size: {app_config.embedding.batch_size}")
        click.echo(f"  Dimensions: {app_config.embedding.dimensions}")
        click.echo(f"  API key set: {'Yes' if app_config.embedding.api_key else 'No'}")

        click.echo("\n[Milvus Configuration]")
        click.echo(f"  Host: {app_config.milvus.host}")
        click.echo(f"  Port: {app_config.milvus.port}")
        click.echo(f"  Collection: {app_config.milvus.collection}")
        click.echo(f"  Index type: {app_config.milvus.index_type}")
        click.echo(f"  Metric type: {app_config.milvus.metric_type}")

        click.echo("\n[Chunking Configuration]")
        click.echo(f"  Chunk size: {app_config.chunking.chunk_size} tokens")
        click.echo(f"  Chunk overlap: {app_config.chunking.chunk_overlap} tokens")
        click.echo(f"  Method: {app_config.chunking.method}")

        click.echo("\n[Retrieval Configuration]")
        click.echo(f"  Hybrid top-k: {app_config.retrieval.hybrid_top_k}")
        click.echo(f"  Rerank top-k: {app_config.retrieval.rerank_top_k}")
        click.echo(f"  RRF k: {app_config.retrieval.rrf_k}")
        click.echo(f"  RRF alpha: {app_config.retrieval.rrf_alpha}")

        click.echo("\n[Reranker Configuration]")
        click.echo(f"  Model: {app_config.reranker.model}")
        click.echo(f"  Device: {app_config.reranker.device}")
        click.echo(f"  Max length: {app_config.reranker.max_length}")

        click.echo("\n[LLM Configuration]")
        click.echo(f"  Model: {app_config.llm.model}")
        click.echo(f"  Temperature: {app_config.llm.temperature}")
        click.echo(f"  Max tokens: {app_config.llm.max_tokens}")
        click.echo(f"  API key set: {'Yes' if app_config.llm.api_key else 'No'}")

        click.echo("\n[OCR Configuration]")
        click.echo(f"  Language: {app_config.ocr.lang}")
        click.echo(f"  Use GPU: {app_config.ocr.use_gpu}")
        click.echo(f"  Confidence threshold: {app_config.ocr.confidence_threshold}")

        click.echo("\n[Paths]")
        click.echo(f"  Data directory: {app_config.paths.data_dir}")
        click.echo(f"  Raw text: {app_config.paths.raw_text_dir}")
        click.echo(f"  Cleaned chunks: {app_config.paths.cleaned_chunks_dir}")
        click.echo(f"  Embedded chunks: {app_config.paths.emb_chunks_dir}")
        click.echo(f"  Logs: {app_config.paths.logs_dir}")

        # Validate API keys
        missing_keys = app_config.validate_api_keys()
        if missing_keys:
            click.echo("\n⚠️  WARNING: Missing API keys:")
            for key in missing_keys:
                click.echo(f"  • {key}")
            click.echo("\nPlease set these in your .env file.")

    elif set_value:
        click.echo("\n✗ Configuration updates via CLI not yet implemented.", err=True)
        click.echo("Please edit the .env file directly and restart the application.")

    click.echo("\n✓ Configuration displayed!")
