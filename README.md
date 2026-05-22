# Document Indexing Assignment

## Project Purpose

This project indexes one `.pdf` or `.docx` document into PostgreSQL.

For each run, the script:
- extracts text from the input file
- applies minimal cleanup
- splits the text with one selected strategy
- generates one Gemini embedding per chunk
- stores chunk text and embeddings in PostgreSQL

The project is intentionally small and explicit. The main code lives in one script:
- `index_documents.py`

## Installation

Use Python `3.11` or later.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## .env Setup

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
POSTGRES_URL=postgresql://username:password@localhost:5432/your_database
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

Required variables:
- `GEMINI_API_KEY`
- `POSTGRES_URL`

Optional variable:
- `GEMINI_EMBEDDING_MODEL`

If `GEMINI_EMBEDDING_MODEL` is not set, the code uses:
- `gemini-embedding-001`

## Database Setup

The database schema is defined in:
- `schema.sql`

Apply it with a PostgreSQL client before running the script. Example with `psql`:

```bash
psql "postgresql://username:password@localhost:5432/your_database" -f schema.sql
```

The table stores these columns:
- `id`: row identifier
- `chunk_text`: the stored chunk text
- `embedding`: the embedding vector as `DOUBLE PRECISION[]`
- `filename`: base filename only, for example `report.pdf`
- `split_strategy`: the strategy used for the current run
- `created_at`: insertion timestamp

## CLI Usage

Command shape:

```bash
python index_documents.py <file_path> --split-strategy <strategy>
```

Arguments:
- positional `file_path`: input `.pdf` or `.docx` file
- required `--split-strategy`: one of `fixed`, `sentence`, `paragraph`
- optional `--chunk-size`: token budget for all strategies, default `512`
- optional `--chunk-overlap`: only for `fixed`, default `64`
- optional `--replace-existing`: delete old rows for the same `filename` and `split_strategy` before insert

## Split Strategies

### fixed

The script splits text by token count.

Behavior:
- default chunk size is `512`
- default overlap is `64`
- `--chunk-size` overrides the chunk size
- `--chunk-overlap` overrides the overlap

### sentence

The script splits text into sentences with regex punctuation boundaries, then groups consecutive sentences until the token budget is reached.

Behavior:
- default chunk size is `512`
- `--chunk-size` overrides the chunk size
- if one sentence is larger than the token budget, it falls back to fixed token-based splitting for that sentence

### paragraph

The script splits text on blank lines, then groups consecutive paragraphs until the token budget is reached.

Behavior:
- default chunk size is `512`
- `--chunk-size` overrides the chunk size
- if one paragraph is larger than the token budget, it falls back to fixed token-based splitting for that paragraph

## Replace Existing

Default behavior is append.

That means a normal run adds new rows even if rows already exist for the same file and strategy.

If you pass `--replace-existing`, the script:
1. deletes rows where `filename` matches the current base filename
2. deletes only rows where `split_strategy` matches the current run
3. inserts the new rows after the delete

## Example Commands

```bash
python index_documents.py sample.pdf --split-strategy fixed
python index_documents.py sample.pdf --split-strategy fixed --chunk-size 400 --chunk-overlap 32
python index_documents.py sample.pdf --split-strategy sentence --replace-existing
python index_documents.py sample.docx --split-strategy paragraph --chunk-size 300
```

## Runtime Behavior

The script runs in this order:
1. load configuration from `.env`
2. validate CLI arguments
3. validate the input path and file type
4. extract text
5. clean the extracted text
6. split text into chunks
7. generate embeddings for all chunks
8. connect to PostgreSQL
9. optionally delete old rows for the same filename and strategy
10. insert all rows in one transaction
11. commit once at the end

Progress messages are printed for the major phases, followed by a final success summary with:
- `filename`
- `split_strategy`
- `chunk_count`
- `inserted_row_count`

## Running Tests

Run the full test suite with:

```bash
python -m unittest discover -s tests -v
```
