"""Setup script for My Atlas RAG Chatbot."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="my-atlas",
    version="1.0.0",
    description="RAG-based chatbot with document knowledge base using hybrid search and reranking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="My Atlas Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "click>=8.1.7",
        "pydantic>=2.5.0",
        "python-json-logger>=2.0.7",
        "PyMuPDF>=1.23.8",
        "python-docx>=1.1.0",
        "paddleocr>=2.7.3",
        "langchain>=0.1.0",
        "langchain-text-splitters>=0.0.1",
        "tiktoken>=0.5.2",
        "nltk>=3.8.1",
        "rank-bm25>=0.2.2",
        "voyageai>=0.2.0",
        "pymilvus>=2.3.4",
        "transformers>=4.36.0",
        "torch>=2.1.2",
        "sentencepiece>=0.1.99",
        "openai>=1.6.1",
        "tqdm>=4.66.1",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "my-atlas=src.cli.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
