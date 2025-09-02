# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/beignet/` (operators live in underscore-prefixed modules, e.g., `_foo.py`).
- Public API: re-export in `src/beignet/__init__.py`.
- Tests: `tests/beignet/` mirrors source (test files like `test__foo.py`).
- Benchmarks: `benchmarks/` (ASV suites for operators and features).
- Docs: `docs/` with MkDocs config in `mkdocs.yml`.

## Build, Test, and Development Commands
- Sync deps: `uv sync` (preferred). Alt: `pip install -e '.[all]'`.
- Run tests: `uv run python -m pytest`.
- Lint: `uv run ruff check` (auto-fix: `uv run ruff check --fix`).
- Format: `uv run ruff format`.
- Build package: `uv run python -m build .`.
- Benchmarks: `uv run asv run` (see Benchmarking).
- Docs: `mkdocs serve` (local), `mkdocs gh-deploy --force` (deploy).

## Coding Style & Naming Conventions
- Python, 4-space indentation, comprehensive type hints.
- One operator per file with underscore prefix (e.g., `_apply_quaternion.py`).
- Batch-first tensor APIs; prefer pure functional style.
- Must work with `torch.compile(fn, fullgraph=True)` and `torch.func` (vmap/grad).
- Lint/format via Ruff; pre-commit hooks provided.

## Testing Guidelines
- Frameworks: `pytest`, `hypothesis` for property-based tests.
- Location: `tests/beignet/` matching source layout.
- Pattern: a focused test module per operator (e.g., `test__foo.py`).
- Validate gradients (`torch.autograd.gradcheck`) and compilation compatibility when relevant.
- Run: `uv run python -m pytest`.

## Benchmarking & Performance
- Tooling: ASV suites in `benchmarks/` with time_*/peak_memory_* metrics.
- Common commands: `uv run asv run`, `uv run asv continuous HEAD~1 HEAD`, `uv run asv publish`, `uv run asv show`.
- Reproducibility: set `BEIGNET_BENCHMARK_SEED` (default 42).

## Documentation
- NumPy-style docstrings with examples and shapes.
- Add new operators to docs under `docs/reference/operators/` by category.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope; group logical changes; run lint/format/tests.
- PRs: clear description, linked issues, tests/benchmarks for new ops, update docs as needed; include performance notes if behavior or speed changes.
