"""Clear command to reset the knowledge base."""

import click
import shutil
from ...database.milvus_client import MilvusClient
from ...utils.config import config
from ...utils.logger import get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    '--confirm',
    is_flag=True,
    help='Skip confirmation prompt'
)
def clear(confirm: bool):
    """Clear the knowledge base and all cached data.

    WARNING: This will delete all documents, chunks, embeddings, and the vector database.

    Examples:
        my-atlas clear
        my-atlas clear --confirm
    """
    click.echo("=" * 60)
    click.echo("MY ATLAS - Clear Knowledge Base")
    click.echo("=" * 60)

    if not confirm:
        click.echo("\n⚠️  WARNING: This will delete:")
        click.echo("  • All raw extracted text")
        click.echo("  • All processed chunks")
        click.echo("  • All embeddings")
        click.echo("  • All data in the vector database")
        click.echo("\nThis action cannot be undone!")

        if not click.confirm("\nAre you sure you want to continue?"):
            click.echo("\nOperation cancelled.")
            return

    try:
        click.echo("\nClearing knowledge base...")

        # Clear Milvus database
        click.echo("  [1/3] Clearing vector database...")
        db_client = MilvusClient()
        db_client.delete_all()
        click.echo("    ✓ Vector database cleared")

        # Clear raw text
        click.echo("  [2/3] Clearing raw text files...")
        if config.paths.raw_text_dir.exists():
            shutil.rmtree(config.paths.raw_text_dir)
            config.paths.raw_text_dir.mkdir(parents=True, exist_ok=True)
        click.echo("    ✓ Raw text cleared")

        # Clear chunks and embeddings
        click.echo("  [3/3] Clearing chunks and embeddings...")

        if config.paths.cleaned_chunks_dir.exists():
            shutil.rmtree(config.paths.cleaned_chunks_dir)
            config.paths.cleaned_chunks_dir.mkdir(parents=True, exist_ok=True)

        if config.paths.emb_chunks_dir.exists():
            shutil.rmtree(config.paths.emb_chunks_dir)
            config.paths.emb_chunks_dir.mkdir(parents=True, exist_ok=True)

        click.echo("    ✓ Chunks and embeddings cleared")

        click.echo("\n✓ Knowledge base cleared successfully!")

    except Exception as e:
        click.echo(f"\n✗ Error during clear operation: {e}", err=True)
        logger.error(f"Clear error: {e}", exc_info=True)
        raise click.Abort()
