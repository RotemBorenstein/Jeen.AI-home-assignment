"""CLI entry point for document indexing.

This module currently implements the early pipeline stages:
- CLI parsing
- CLI validation
- environment loading
- input file validation

Later steps will add extraction, chunking, embeddings, and database writes.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable


DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"
CL100K_BASE_FILE_PATH = Path(__file__).resolve().parent / "vendor" / "cl100k_base.tiktoken"
CL100K_BASE_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)
CL100K_BASE_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
SPLIT_STRATEGIES = ("fixed", "sentence", "paragraph")


class CliValidationError(ValueError):
    """Raised when parsed CLI arguments use an invalid combination."""


class SettingsError(ValueError):
    """Raised when required configuration values are missing."""


class InputValidationError(ValueError):
    """Raised when the input file path or file type is invalid."""


class ExtractionError(ValueError):
    """Raised when a supported document cannot be parsed into text."""


class TokenizationError(ValueError):
    """Raised when token counting cannot initialize the configured encoding."""


class EmbeddingError(ValueError):
    """Raised when Gemini embedding generation fails or returns a bad shape."""


class DatabaseError(ValueError):
    """Raised when database setup or write preparation fails."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from the environment."""

    gemini_api_key: str
    postgres_url: str
    gemini_embedding_model: str


@dataclass(frozen=True)
class ValidatedInputFile:
    """Validated input file metadata used by the indexing flow."""

    file_path: Path
    filename: str
    extension: str


@dataclass(frozen=True)
class ChunkEmbeddingRecord:
    """A chunk and its embedding, ready for database insertion."""

    chunk_text: str
    embedding: list[float]


@dataclass(frozen=True)
class InsertionSummary:
    """Insertion result details used for the final success output."""

    inserted_row_count: int


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the document indexing script."""

    parser = argparse.ArgumentParser(
        description="Index one PDF or DOCX document into PostgreSQL with Gemini embeddings."
    )
    parser.add_argument("file_path", help="Path to the input PDF or DOCX file.")
    parser.add_argument(
        "--split-strategy",
        required=True,
        choices=SPLIT_STRATEGIES,
        help="Chunking strategy to use: fixed, sentence, or paragraph.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Token budget for each chunk. Defaults to 512.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Token overlap between fixed-size chunks. Defaults to 64 for the fixed strategy.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete existing rows for the same filename and split strategy before insert.",
    )
    return parser


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse raw CLI arguments into a namespace."""

    parser = build_argument_parser()
    return parser.parse_args(argv)


def validate_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate CLI argument combinations after parsing."""

    if args.chunk_overlap is not None and args.split_strategy != "fixed":
        raise CliValidationError(
            "--chunk-overlap can only be used with --split-strategy fixed."
        )
    return args


def load_settings() -> Settings:
    """Load required settings from .env and process environment variables."""

    from dotenv import load_dotenv

    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    postgres_url = os.getenv("POSTGRES_URL")
    gemini_embedding_model = os.getenv(
        "GEMINI_EMBEDDING_MODEL", DEFAULT_GEMINI_EMBEDDING_MODEL
    )

    missing_variables = []
    if not gemini_api_key:
        missing_variables.append("GEMINI_API_KEY")
    if not postgres_url:
        missing_variables.append("POSTGRES_URL")

    if missing_variables:
        missing_list = ", ".join(missing_variables)
        raise SettingsError(f"Missing required environment variables: {missing_list}")

    return Settings(
        gemini_api_key=gemini_api_key,
        postgres_url=postgres_url,
        gemini_embedding_model=gemini_embedding_model,
    )


def validate_input_file(file_path: str) -> ValidatedInputFile:
    """Validate that the input path exists and has a supported extension."""

    resolved_path = Path(file_path)
    if not resolved_path.exists():
        raise InputValidationError(f"Input file does not exist: {resolved_path}")

    extension = resolved_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported_values = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise InputValidationError(
            f"Unsupported file extension '{resolved_path.suffix}'. Supported extensions: {supported_values}"
        )

    return ValidatedInputFile(
        file_path=resolved_path,
        filename=resolved_path.name,
        extension=extension,
    )


def extract_docx_text(file_path: Path) -> str:
    """Extract text from a DOCX file by joining paragraphs in document order."""

    from docx import Document

    try:
        document = Document(file_path)
    except Exception as error:
        raise ExtractionError(f"Failed to parse DOCX file '{file_path}': {error}") from error

    paragraph_texts = [paragraph.text for paragraph in document.paragraphs if paragraph.text != ""]
    return "\n\n".join(paragraph_texts)


def normalize_pdf_page_text(page_text: str) -> str:
    """Merge wrapped PDF lines into cleaner paragraph text for downstream chunking."""

    normalized_text = page_text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = normalized_text.split("\n")

    paragraphs: list[str] = []
    current_parts: list[str] = []

    def flush_current_parts() -> None:
        if current_parts:
            paragraphs.append(" ".join(current_parts))
            current_parts.clear()

    for line_index, raw_line in enumerate(raw_lines):
        line = re.sub(r"[ \t]{2,}", " ", raw_line).strip()

        if not line:
            flush_current_parts()
            continue

        next_non_empty_line = get_next_non_empty_pdf_line(raw_lines, line_index + 1)
        if is_pdf_heading_line(line, next_non_empty_line=next_non_empty_line):
            flush_current_parts()
            paragraphs.append(line)
            continue

        if line.startswith("•"):
            flush_current_parts()
            current_parts.append(line)
            continue

        if current_parts and should_merge_pdf_without_space(current_parts[-1], line):
            if current_parts[-1].endswith(" -"):
                current_parts[-1] = current_parts[-1][:-2] + "-" + line
            else:
                current_parts[-1] = current_parts[-1].rstrip() + line
            continue

        current_parts.append(line)

    flush_current_parts()
    return "\n\n".join(paragraphs)


def get_next_non_empty_pdf_line(raw_lines: list[str], start_index: int) -> str | None:
    """Return the next non-empty PDF line after the current line, if one exists."""

    for raw_line in raw_lines[start_index:]:
        normalized_line = re.sub(r"[ \t]{2,}", " ", raw_line).strip()
        if normalized_line:
            return normalized_line
    return None


def is_pdf_heading_line(line: str, next_non_empty_line: str | None = None) -> bool:
    """Detect short section-title lines that should remain standalone paragraphs."""

    if len(line) > 80:
        return False
    if line.startswith("•"):
        return False
    if re.search(r"[.!?;:]$", line):
        return False

    words = line.split()
    if not words or len(words) > 4:
        return False

    if not re.match(r"^[A-Z]", words[0]):
        return False

    if next_non_empty_line and next_non_empty_line[:1].islower():
        return False

    return all(re.match(r"^[A-Za-z0-9'&()-]+$", word) for word in words)


def should_merge_pdf_without_space(previous_text: str, next_line: str) -> bool:
    """Rejoin a PDF line-wrap split that broke a hyphenated word across lines."""

    return previous_text.endswith("-") or previous_text.endswith(" -")


def extract_pdf_text(file_path: Path) -> str:
    """Extract text from a PDF file by joining page text with double newlines."""

    from pypdf import PdfReader

    try:
        reader = PdfReader(file_path)
        page_texts = []
        for page in reader.pages:
            page_text = page.extract_text()
            page_texts.append(normalize_pdf_page_text(page_text or ""))
    except Exception as error:
        raise ExtractionError(f"Failed to parse PDF file '{file_path}': {error}") from error

    return "\n\n".join(page_texts)


def extract_text_from_file(input_file: ValidatedInputFile) -> str:
    """Dispatch the validated file to the correct text extractor."""

    if input_file.extension == ".docx":
        extracted_text = extract_docx_text(input_file.file_path)
    elif input_file.extension == ".pdf":
        extracted_text = extract_pdf_text(input_file.file_path)
    else:
        raise ExtractionError(
            f"Unsupported file extension reached extractor: {input_file.extension}"
        )

    if extracted_text:
        return extracted_text

    raise ExtractionError(f"No text could be extracted from '{input_file.file_path}'.")


def clean_extracted_text(text: str) -> str:
    """Normalize line endings, trim edges, and collapse repeated blank lines."""

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_text = normalized_text.strip()
    return re.sub(r"\n{3,}", "\n\n", normalized_text)


@lru_cache(maxsize=None)
def load_local_cl100k_base_encoding():
    """Build the vendored cl100k_base encoding from the local repository file.

    Git can rewrite local line endings in a fresh clone. The BPE parser still
    understands the file, but strict blob-hash verification can reject it.
    """

    import tiktoken
    import tiktoken.load

    mergeable_ranks = tiktoken.load.load_tiktoken_bpe(str(CL100K_BASE_FILE_PATH))
    return tiktoken.Encoding(
        name=DEFAULT_TIKTOKEN_ENCODING,
        pat_str=CL100K_BASE_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens=CL100K_BASE_SPECIAL_TOKENS,
    )


@lru_cache(maxsize=None)
def get_token_encoding(encoding_name: str = DEFAULT_TIKTOKEN_ENCODING):
    """Load and cache the configured tiktoken encoding."""

    import tiktoken

    try:
        if encoding_name == DEFAULT_TIKTOKEN_ENCODING:
            return load_local_cl100k_base_encoding()
        return tiktoken.get_encoding(encoding_name)
    except Exception as error:
        raise TokenizationError(
            f"Failed to load tiktoken encoding '{encoding_name}': {error}"
        ) from error


def count_text_tokens(text: str, encoding_name: str = DEFAULT_TIKTOKEN_ENCODING) -> int:
    """Count text tokens with a local tiktoken encoding approximation."""

    if not text:
        return 0

    encoding = get_token_encoding(encoding_name)
    return len(encoding.encode(text))


def chunk_text_fixed(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
) -> list[str]:
    """Split text into token-based chunks with overlap."""

    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    encoding = get_token_encoding(encoding_name)
    tokens = encoding.encode(text)
    if not tokens:
        return []

    chunk_texts = []
    start_index = 0
    step_size = chunk_size - chunk_overlap

    while start_index < len(tokens):
        end_index = start_index + chunk_size
        chunk_tokens = tokens[start_index:end_index]
        chunk_texts.append(encoding.decode(chunk_tokens))
        if end_index >= len(tokens):
            break
        start_index += step_size

    return chunk_texts


def split_text_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex punctuation boundaries."""

    stripped_text = text.strip()
    if not stripped_text:
        return []

    return re.split(r"(?<=[.!?])\s+", stripped_text)


def chunk_text_by_sentences(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
) -> list[str]:
    """Group consecutive sentences to a token budget with fixed-chunk fallback."""

    sentences = split_text_into_sentences(text)
    return _group_text_units_with_fallback(
        text_units=sentences,
        chunk_size=chunk_size,
        encoding_name=encoding_name,
        join_separator=" ",
    )


def split_text_into_paragraphs(text: str) -> list[str]:
    """Split normalized text into paragraphs on blank lines."""

    stripped_text = text.strip()
    if not stripped_text:
        return []

    return re.split(r"\n\s*\n", stripped_text)


def chunk_text_by_paragraphs(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
) -> list[str]:
    """Group consecutive paragraphs to a token budget with fixed-chunk fallback."""

    paragraphs = split_text_into_paragraphs(text)
    return _group_text_units_with_fallback(
        text_units=paragraphs,
        chunk_size=chunk_size,
        encoding_name=encoding_name,
        join_separator="\n\n",
    )


def resolve_fallback_chunk_overlap(chunk_size: int) -> int:
    """Clamp fallback overlap so fixed fallback remains valid for any chunk size."""

    if chunk_size <= 1:
        return 0
    return min(DEFAULT_CHUNK_OVERLAP, chunk_size - 1)


def _group_text_units_with_fallback(
    text_units: list[str],
    chunk_size: int,
    encoding_name: str,
    join_separator: str,
) -> list[str]:
    """Build chunk groups from sentences or paragraphs and split oversized units."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    chunks: list[str] = []
    current_units: list[str] = []

    for text_unit in text_units:
        if not text_unit:
            continue

        text_unit_tokens = count_text_tokens(text_unit, encoding_name=encoding_name)
        if text_unit_tokens > chunk_size:
            if current_units:
                chunks.append(join_separator.join(current_units))
                current_units = []

            chunks.extend(
                chunk_text_fixed(
                    text_unit,
                    chunk_size=chunk_size,
                    chunk_overlap=resolve_fallback_chunk_overlap(chunk_size),
                    encoding_name=encoding_name,
                )
            )
            continue

        candidate_units = current_units + [text_unit]
        candidate_text = join_separator.join(candidate_units)
        candidate_tokens = count_text_tokens(candidate_text, encoding_name=encoding_name)

        if candidate_tokens <= chunk_size:
            current_units = candidate_units
            continue

        chunks.append(join_separator.join(current_units))
        current_units = [text_unit]

    if current_units:
        chunks.append(join_separator.join(current_units))

    return chunks


def chunk_text(
    text: str,
    split_strategy: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
) -> list[str]:
    """Dispatch to the selected chunking strategy."""

    resolved_chunk_size = chunk_size or DEFAULT_CHUNK_SIZE

    if split_strategy == "fixed":
        resolved_chunk_overlap = (
            DEFAULT_CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        )
        return chunk_text_fixed(
            text,
            chunk_size=resolved_chunk_size,
            chunk_overlap=resolved_chunk_overlap,
            encoding_name=encoding_name,
        )
    if split_strategy == "sentence":
        return chunk_text_by_sentences(
            text,
            chunk_size=resolved_chunk_size,
            encoding_name=encoding_name,
        )
    if split_strategy == "paragraph":
        return chunk_text_by_paragraphs(
            text,
            chunk_size=resolved_chunk_size,
            encoding_name=encoding_name,
        )

    raise ValueError(f"Unsupported split strategy: {split_strategy}")


def create_gemini_client(api_key: str):
    """Create a Gemini API client for embedding requests."""

    from google import genai

    return genai.Client(api_key=api_key)


def generate_chunk_embedding(
    client,
    model_name: str,
    chunk_text: str,
) -> list[float]:
    """Generate one embedding vector for one chunk of text."""

    try:
        response = client.models.embed_content(model=model_name, contents=chunk_text)
    except Exception as error:
        raise EmbeddingError(f"Failed to generate embedding: {error}") from error

    embeddings = getattr(response, "embeddings", None)
    if not embeddings:
        raise EmbeddingError("Gemini embedding response did not include embeddings.")

    first_embedding = embeddings[0]
    values = getattr(first_embedding, "values", None)
    if values is None:
        raise EmbeddingError("Gemini embedding response did not include embedding values.")

    return list(values)


def generate_chunk_embedding_with_retry(
    client,
    model_name: str,
    chunk_text: str,
    max_attempts: int = 2,
    retry_delay_seconds: int = 1,
) -> list[float]:
    """Generate one embedding and retry once on transient failure."""

    last_error: Exception | None = None

    for attempt_number in range(1, max_attempts + 1):
        try:
            return generate_chunk_embedding(
                client=client,
                model_name=model_name,
                chunk_text=chunk_text,
            )
        except EmbeddingError as error:
            last_error = error
            if attempt_number == max_attempts:
                break
            time.sleep(retry_delay_seconds)

    raise EmbeddingError(
        f"Failed to generate embedding after {max_attempts} attempts: {last_error}"
    ) from last_error


def generate_embeddings_for_chunks(
    chunk_texts: list[str],
    client,
    model_name: str,
) -> list[ChunkEmbeddingRecord]:
    """Generate embeddings for every chunk before any database write begins."""

    chunk_records: list[ChunkEmbeddingRecord] = []

    for chunk_text in chunk_texts:
        embedding = generate_chunk_embedding_with_retry(
            client=client,
            model_name=model_name,
            chunk_text=chunk_text,
        )
        chunk_records.append(
            ChunkEmbeddingRecord(chunk_text=chunk_text, embedding=embedding)
        )

    return chunk_records


def connect_to_postgres(postgres_url: str):
    """Create a PostgreSQL connection from the configured URL."""

    import psycopg

    try:
        return psycopg.connect(postgres_url)
    except Exception as error:
        raise DatabaseError(f"Failed to connect to PostgreSQL: {error}") from error


def validate_database_schema(connection) -> None:
    """Fail early if the required table is missing from the target database."""

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM document_chunks LIMIT 1")
    except Exception as error:
        raise DatabaseError(
            "PostgreSQL is reachable, but the required table 'document_chunks' is missing. "
            "Apply schema.sql to the configured database before running the indexer."
        ) from error


def delete_existing_rows(connection, filename: str, split_strategy: str) -> None:
    """Delete rows for one filename and split strategy."""

    delete_query = """
        DELETE FROM document_chunks
        WHERE filename = %s AND split_strategy = %s
    """

    with connection.cursor() as cursor:
        cursor.execute(delete_query, (filename, split_strategy))


def run_replace_existing_if_requested(
    connection,
    filename: str,
    split_strategy: str,
    replace_existing: bool,
    insert_operation=None,
) -> None:
    """Run optional replacement before the later insert step."""

    if replace_existing:
        delete_existing_rows(connection, filename, split_strategy)

    if insert_operation is not None:
        insert_operation()


def build_insert_rows(
    chunk_records: list[ChunkEmbeddingRecord],
    filename: str,
    split_strategy: str,
) -> list[tuple[str, list[float], str, str]]:
    """Build explicit row tuples for database insertion."""

    return [
        (record.chunk_text, record.embedding, filename, split_strategy)
        for record in chunk_records
    ]


def insert_chunk_rows(
    connection,
    chunk_records: list[ChunkEmbeddingRecord],
    filename: str,
    split_strategy: str,
) -> InsertionSummary:
    """Insert all chunk rows in one transaction and commit once."""

    insert_query = """
        INSERT INTO document_chunks (chunk_text, embedding, filename, split_strategy)
        VALUES (%s, %s, %s, %s)
    """
    insert_rows = build_insert_rows(chunk_records, filename, split_strategy)

    try:
        with connection.cursor() as cursor:
            for insert_row in insert_rows:
                cursor.execute(insert_query, insert_row)
        connection.commit()
    except Exception as error:
        raise DatabaseError(f"Failed to insert document chunks: {error}") from error

    return InsertionSummary(inserted_row_count=len(insert_rows))


def print_progress(message: str) -> None:
    """Print one concise progress message."""

    print(message)


def print_success_summary(
    filename: str,
    split_strategy: str,
    chunk_count: int,
    inserted_row_count: int,
) -> None:
    """Print the final success summary required by the PRD."""

    print(
        "Indexing completed successfully. "
        f"filename={filename} "
        f"split_strategy={split_strategy} "
        f"chunk_count={chunk_count} "
        f"inserted_row_count={inserted_row_count}"
    )


def run_indexing_pipeline(
    args: argparse.Namespace,
    *,
    progress_printer: Callable[[str], None] = print_progress,
) -> InsertionSummary:
    """Run the full indexing pipeline in explicit sequential stages."""

    progress_printer("Loading configuration...")
    settings = load_settings()

    progress_printer("Validating input file...")
    input_file = validate_input_file(args.file_path)

    progress_printer("Extracting text...")
    extracted_text = extract_text_from_file(input_file)

    cleaned_text = clean_extracted_text(extracted_text)

    progress_printer("Splitting text into chunks...")
    chunks = chunk_text(
        cleaned_text,
        split_strategy=args.split_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    progress_printer("Validating PostgreSQL setup...")
    connection = connect_to_postgres(settings.postgres_url)
    try:
        validate_database_schema(connection)

        progress_printer("Generating embeddings...")
        gemini_client = create_gemini_client(settings.gemini_api_key)
        chunk_records = generate_embeddings_for_chunks(
            chunks,
            client=gemini_client,
            model_name=settings.gemini_embedding_model,
        )

        progress_printer("Writing rows to PostgreSQL...")
        insertion_summary = run_replace_and_insert(
            connection=connection,
            filename=input_file.filename,
            split_strategy=args.split_strategy,
            replace_existing=args.replace_existing,
            chunk_records=chunk_records,
        )
    finally:
        close_method = getattr(connection, "close", None)
        if callable(close_method):
            close_method()

    print_success_summary(
        filename=input_file.filename,
        split_strategy=args.split_strategy,
        chunk_count=len(chunks),
        inserted_row_count=insertion_summary.inserted_row_count,
    )
    return insertion_summary


def run_replace_and_insert(
    connection,
    filename: str,
    split_strategy: str,
    replace_existing: bool,
    chunk_records: list[ChunkEmbeddingRecord],
) -> InsertionSummary:
    """Run optional delete, then insert rows for this indexing run."""

    insertion_result: InsertionSummary | None = None

    def insert_operation() -> None:
        nonlocal insertion_result
        insertion_result = insert_chunk_rows(
            connection=connection,
            chunk_records=chunk_records,
            filename=filename,
            split_strategy=split_strategy,
        )

    run_replace_existing_if_requested(
        connection=connection,
        filename=filename,
        split_strategy=split_strategy,
        replace_existing=replace_existing,
        insert_operation=insert_operation,
    )

    if insertion_result is None:
        raise DatabaseError("Insert operation did not run.")

    return insertion_result


def main(argv: list[str] | None = None) -> int:
    """Run the full document indexing CLI flow."""

    args = parse_cli_args(argv)
    validate_cli_args(args)
    run_indexing_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
