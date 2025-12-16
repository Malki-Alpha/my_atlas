"""Ingest command for document ingestion."""

import click
from pathlib import Path
from ...ingestion.pipeline import IngestionPipeline
from ...utils.logger import get_logger

logger = get_logger(__name__)


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option(
    '--doc-type',
    type=click.Choice(['pdf', 'docx', 'all'], case_sensitive=False),
    default='all',
    help='Type of documents to process'
)
@click.option(
    '--use-ocr',
    is_flag=True,
    help='Use OCR for PDF extraction (for scanned documents)'
)
@click.option(
    '--batch-size',
    type=int,
    default=None,
    help='Batch size for processing (not yet implemented)'
)
def ingest(path: str, doc_type: str, use_ocr: bool, batch_size: int):
    """Ingest documents into the knowledge base.

    PATH: Path to document file or directory containing documents

    Examples:
        my-atlas ingest ./data/documents
        my-atlas ingest ./data/mydoc.pdf --use-ocr
        my-atlas ingest ./data/documents --doc-type pdf
    """
    click.echo("=" * 60)
    click.echo("MY ATLAS - Document Ingestion")
    click.echo("=" * 60)

    path_obj = Path(path)

    # Initialize pipeline
    click.echo(f"\nInitializing ingestion pipeline (OCR: {use_ocr})...")
    pipeline = IngestionPipeline(use_ocr=use_ocr)

    try:
        if path_obj.is_file():
            # Ingest single file
            click.echo(f"\nIngesting file: {path_obj.name}")

            with click.progressbar(length=1, label='Processing') as bar:
                result = pipeline.ingest_document(path_obj)
                bar.update(1)

            # Display results
            click.echo("\nIngestion Results:")
            click.echo(f"  Document: {result['document']}")
            click.echo(f"  Status: {result['status']}")
            click.echo(f"  Pages extracted: {result['pages_extracted']}")
            click.echo(f"  Chunks created: {result['chunks_created']}")
            click.echo(f"  Chunks embedded: {result['chunks_embedded']}")
            click.echo(f"  Chunks inserted: {result['chunks_inserted']}")
            click.echo(f"  Processing time: {result['processing_time']:.2f}s")

            if result['status'] == 'error':
                click.echo(f"  Error: {result['error']}", err=True)

        elif path_obj.is_dir():
            # Ingest directory
            click.echo(f"\nIngesting documents from: {path_obj}")
            click.echo(f"Document type filter: {doc_type}")

            results = pipeline.ingest_directory(path_obj, doc_type=doc_type)

            if not results:
                click.echo("\nNo documents found to ingest.", err=True)
                return

            # Display summary
            total_docs = len(results)
            successful = sum(1 for r in results if r['status'] == 'success')
            failed = total_docs - successful
            total_chunks = sum(r['chunks_inserted'] for r in results)
            total_time = sum(r['processing_time'] for r in results)

            click.echo("\n" + "=" * 60)
            click.echo("INGESTION SUMMARY")
            click.echo("=" * 60)
            click.echo(f"Total documents: {total_docs}")
            click.echo(f"Successful: {successful}")
            click.echo(f"Failed: {failed}")
            click.echo(f"Total chunks inserted: {total_chunks}")
            click.echo(f"Total processing time: {total_time:.2f}s")
            click.echo(f"Average time per document: {total_time/total_docs:.2f}s")

            # Show failed documents if any
            if failed > 0:
                click.echo("\nFailed documents:")
                for r in results:
                    if r['status'] == 'error':
                        click.echo(f"  - {r['document']}: {r['error']}")

        click.echo("\n✓ Ingestion complete!")

    except Exception as e:
        click.echo(f"\n✗ Error during ingestion: {e}", err=True)
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise click.Abort()
