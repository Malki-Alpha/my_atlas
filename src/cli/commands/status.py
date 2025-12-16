"""Status command to show system information."""

import click
from ...database.milvus_client import MilvusClient
from ...ingestion.pipeline import IngestionPipeline
from ...utils.config import config
from ...utils.logger import get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    '--docs',
    is_flag=True,
    help='Show detailed document statistics'
)
@click.option(
    '--chunks',
    is_flag=True,
    help='Show detailed chunk statistics'
)
@click.option(
    '--db',
    is_flag=True,
    help='Show database health information'
)
def status(docs: bool, chunks: bool, db: bool):
    """Show system status and statistics.

    Examples:
        my-atlas status
        my-atlas status --docs
        my-atlas status --db
    """
    click.echo("=" * 60)
    click.echo("MY ATLAS - System Status")
    click.echo("=" * 60)

    try:
        # Database status
        click.echo("\n[Database Status]")
        db_client = MilvusClient()
        db_status = db_client.get_status()

        if db_status['connected']:
            click.echo(f"  Status: ✓ Connected")
            click.echo(f"  Server: {db_status['host']}:{db_status['port']}")
            click.echo(f"  Version: {db_status['server_version']}")

            if db_status['collection_exists']:
                click.echo(f"  Collection: {db_status['collection_name']}")
                click.echo(f"  Total chunks: {db_status['total_chunks']}")
            else:
                click.echo(f"  Collection: Not created yet")
        else:
            click.echo(f"  Status: ✗ Not connected")
            click.echo(f"  Error: {db_status.get('error', 'Unknown')}")

        # Pipeline statistics
        if docs or chunks or not any([docs, chunks, db]):
            click.echo("\n[Ingestion Pipeline]")
            pipeline = IngestionPipeline()
            stats = pipeline.get_stats()

            click.echo(f"  Raw text files: {stats['raw_text_files']}")
            click.echo(f"  Chunk files: {stats['chunk_files']}")
            click.echo(f"  Embedded chunks: {stats['embedded_files']}")
            click.echo(f"  Database chunks: {stats['database_chunks']}")

        # Configuration
        if not any([docs, chunks, db]):
            click.echo("\n[Configuration]")
            click.echo(f"  Embedding model: {config.embedding.model}")
            click.echo(f"  LLM model: {config.llm.model}")
            click.echo(f"  Reranker model: {config.reranker.model}")
            click.echo(f"  Chunk size: {config.chunking.chunk_size} tokens")
            click.echo(f"  Overlap: {config.chunking.chunk_overlap} tokens")

        # Detailed database info
        if db and db_status['connected']:
            click.echo("\n[Database Details]")
            click.echo(f"  Index type: {config.milvus.index_type}")
            click.echo(f"  Metric type: {config.milvus.metric_type}")
            click.echo(f"  HNSW M: {config.milvus.hnsw_m}")
            click.echo(f"  HNSW ef: {config.milvus.hnsw_ef_search}")

        click.echo("\n✓ Status check complete!")

    except Exception as e:
        click.echo(f"\n✗ Error checking status: {e}", err=True)
        logger.error(f"Status error: {e}", exc_info=True)
        raise click.Abort()
