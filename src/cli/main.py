"""Main CLI entry point for My Atlas RAG Chatbot."""

import click
from pathlib import Path
from .commands import ingest, query, status, clear, config_cmd
from ..utils.logger import get_logger

logger = get_logger(__name__)


@click.group()
@click.version_option(version="1.0.0", prog_name="my-atlas")
def cli():
    """My Atlas - RAG-based chatbot with document knowledge base.

    A CLI tool for ingesting documents and querying them using advanced
    RAG techniques with hybrid search and reranking.
    """
    pass


# Register commands
cli.add_command(ingest.ingest)
cli.add_command(query.query)
cli.add_command(status.status)
cli.add_command(clear.clear)
cli.add_command(config_cmd.config)


if __name__ == "__main__":
    cli()
