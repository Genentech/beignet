# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Beignet is a standard library for biological research built on PyTorch. It provides specialized operators for computational biology, chemistry, and physics calculations including geometric transformations, orthogonal polynomials, special functions, and molecular analysis tools.

## Development Commands

### Essential Commands
- **Sync dependencies**: `uv sync` (preferred package manager)
- **Run tests**: `uv run python -m pytest`
- **Lint code**: `uv run ruff check` (with auto-fix: `uv run ruff check --fix`)
- **Format code**: `uv run ruff format`
- **Build package**: `uv run python -m build .`
- **Run benchmarks**: `uv run asv run` (see Benchmarking section for details)

### Development Setup
- **Sync all dependencies**: `uv sync` (installs all optional groups automatically)
- **Alternative install for development**: `python -m pip install --editable '.[all]'`
- **Install test dependencies**: `python -m pip install --editable '.[test]'`
- **Install docs dependencies**: `python -m pip install --editable '.[docs]'`

### Benchmarking
- **Run all benchmarks**: `uv run asv run`
- **Run specific benchmarks**: `uv run asv run -b bench_foo`
- **Compare commits**: `uv run asv continuous HEAD~1 HEAD`
- **Generate HTML reports**: `uv run asv publish`
- **View results**: `uv run asv show`

### Documentation
- **Serve docs locally**: `mkdocs serve`
- **Deploy docs**: `mkdocs gh-deploy --force`

## Architecture

### Core Structure
- **src/beignet/**: Main package with ~200+ mathematical and scientific functions
- **beignet.datasets/**: PyTorch datasets for biological and chemical data
- **beignet.features/**: Feature extraction for geometric transformations
- **beignet.structure/**: Protein structure analysis and manipulation
- **beignet.special/**: Special mathematical functions (error functions, integrals)
- **beignet.constants/**: Biological constants and lookup tables

### Key Functional Areas
1. **Geometric Transformations**: Euler angles, quaternions, rotation matrices, transforms
2. **Orthogonal Polynomials**: Chebyshev, Hermite, Laguerre, Legendre polynomials with full mathematical operations
3. **Molecular Analysis**: Protein structure manipulation, trajectory analysis, contact matrices
4. **Scientific Computing**: Numerical integration, root finding, special functions

### Code Organization
- Each function is implemented in a separate file prefixed with underscore (e.g., `_apply_quaternion.py`)
- Functions are imported and exposed through `__init__.py` files
- Test files mirror the source structure in `tests/beignet/`
- Extensive use of PyTorch tensors and operations throughout

### Testing
- Uses pytest with hypothesis for property-based testing
- Test fixtures handle different dtype scenarios (float32/float64)
- Tests located in `tests/beignet/` mirroring source structure
- Run with `python -m pytest` from project root

### Configuration
- **pyproject.toml**: Contains all project configuration including dependencies, build system, and Ruff linting rules
- **Ruff linting**: Configured to use FLAKE8-BUGBEAR, PYCODESTYLE, PYFLAKES, ISORT rules
- **Pre-commit hooks**: Automated formatting and linting on commits
- **GitHub Actions**: Automated testing on multiple Python versions (3.10-3.12) and platforms

### Dependencies
- **Core**: PyTorch, NumPy 2+, biotite, einops, fastpdb, optree, pooch, tqdm
- **Optional datasets**: biopython, lmdb, pandas for extended dataset support
- **Testing**: pytest, hypothesis, pytest-mock, scipy, psutil (for benchmarking)
- **Docs**: mkdocs-material, mkdocstrings

### Benchmarking System
- **ASV (Airspeed Velocity) benchmark suite** for all operators in `benchmarks/`
- **Comprehensive coverage**: Individual benchmarks for all 216+ operators, datasets, and features
- **Categories**: geometric transformations, polynomials, numerical analysis, molecular operations, special functions  
- **Performance metrics**: execution time (time_*) and memory usage (peak_memory_*)
- **torch.compile optimization**: All operators compiled with fullgraph=True for optimal performance
- **Reproducible results**: Seed management via BEIGNET_BENCHMARK_SEED environment variable (default: 42)
- **Parameterized benchmarks**: Multiple batch sizes and dtypes tested
- **ASV commands**:
  - `uv run asv run`: Run all benchmarks
  - `uv run asv continuous`: Compare performance between commits
  - `uv run asv publish`: Generate HTML reports
  - `uv run asv show`: Display benchmark results

## Adding New Operators

When adding new operators to Beignet, follow these guidelines to ensure consistency and quality:

### 1. Module Structure
- **Location**: Place operators at the root level of the package (e.g., `beignet.foo`, not `beignet.submodule.foo`)
- **File naming**: Create a dedicated module file with underscore prefix (e.g., `_foo.py`)
- **Export**: Add the operator to `src/beignet/__init__.py` to make it publicly available:
  ```python
  from ._foo import foo
  __all__ = [..., "foo"]
  ```

### 2. Implementation Requirements
- **Differentiability**: Ensure operators are differentiable when mathematically sensible. Use PyTorch's autograd-compatible operations and avoid breaking the computation graph
- **Compilation**: Operators must be compatible with `torch.compile(foo, fullgraph=True)`. Avoid Python control flow that depends on tensor values
- **Batch operations**: Design operators to work on batched inputs, following PyTorch conventions. The first dimension should typically be the batch dimension
- **Functional compatibility**: Ensure compatibility with `torch.func` transformations (vmap, grad, etc.) by using pure functional implementations without side effects

### 3. Testing
- **Test file**: Create `tests/beignet/test__foo.py` (note the double underscore to match the source file)
- **Test structure**: Include exactly one test function that comprehensively tests all functionality, integrating property-based testing:
  ```python
  from hypothesis import given, strategies as st
  
  @given(
      batch_size=st.integers(min_value=1, max_value=10),
      dtype=st.sampled_from([torch.float32, torch.float64])
  )
  def test_foo(batch_size, dtype):
      # Test basic functionality
      # Test edge cases
      # Test different dtypes
      # Test batch operations
      # Test mathematical properties hold
      
      # Test gradients with torch.autograd
      # Use torch.autograd.gradcheck to verify correct gradient computation
      
      # Test compatibility with torch.func transformations
      # Verify vmap, grad, and other functional transformations work correctly
      
      # Test torch.compile compatibility
      # Verify the operator can be compiled with torch.compile(fullgraph=True)
  ```
- **Property testing**: The single test function should use Hypothesis decorators to ensure comprehensive coverage across different input scenarios

### 4. Benchmarking
- **Benchmark file**: Create `benchmarks/bench_foo.py` with ASV-compatible benchmark classes
- **Benchmark structure**:
  ```python
  class TimeFoo:
      params = ([1, 10, 100], [torch.float32, torch.float64])
      param_names = ["batch_size", "dtype"]
      
      def setup(self, batch_size, dtype):
          # Setup test data
          
      def time_foo(self, batch_size, dtype):
          # Benchmark the operator
  ```
- **Memory benchmarks**: Include peak memory usage measurements when relevant

### 5. Documentation
- **Docstring format**: Follow NumPy-style docstrings used throughout the project:
  ```python
  def foo(input: Tensor, param: float) -> Tensor:
      r"""
      Short description of the operator.
      
      Parameters
      ----------
      input : Tensor, shape=(..., N)
          Description of input parameter.
      param : float
          Description of parameter.
          
      Returns
      -------
      output : Tensor, shape=(..., M)
          Description of output.
      
      Examples
      --------
      >>> input = torch.randn(10, 3)
      >>> output = beignet.foo(input, 2.0)
      >>> output.shape
      torch.Size([10, 5])
      """
  ```
- **API documentation**: Add the operator to the appropriate section in `docs/reference/operators/`. Create subdirectories as needed based on the operator category (e.g., `geometry/transformations/`, `special-functions/`, etc.)

### 6. Code Quality
- **Type hints**: Include comprehensive type annotations for all parameters and return values
- **Input validation**: Validate input shapes and types, raising appropriate errors
- **Error messages**: Provide clear, actionable error messages
- **Code style**: Follow the project's linting rules (run `uv run ruff check` and `uv run ruff format`)

## Code Style Requirements

### Variable Naming
- **All variables must use lowercase_with_underscores**: Never use PascalCase, camelCase, or UPPER_CASE for variables
- **Descriptive names, no abbreviations**: Use full, descriptive variable names instead of abbreviations for clarity
- **Avoid unnecessary suffixes/prefixes**: Remove redundant words like `_parameter`, `_value`, `_variable` when the context is clear
- **Mathematical variables**: Even single-letter mathematical variables should be lowercase (e.g., `n` not `N`, `r2` not `R2`)
- **Examples**:
  - ✅ `sample_size`, `effect_size`, `degrees_of_freedom`, `noncentrality`, `standard_error`
  - ❌ `sampleSize`, `N`, `R2`, `ncp`, `se`, `df`, `noncentrality_parameter`, `effect_size_value`

### Statement Spacing
- **Add newlines between statements of different line lengths**: When consecutive assignment statements have different character counts, separate them with blank lines for improved readability
- **Example**:
  ```python
  # Correct - newlines between different length statements
  n0 = torch.as_tensor(sample_size)

  groups0 = torch.as_tensor(groups)

  r20 = torch.as_tensor(covariate_r2)

  num_covariates0 = torch.as_tensor(n_covariates)
  
  # Same length statements can be grouped together
  effect_size = effect_size.to(dtype)
  sample_size = sample_size.to(dtype)
  ```

### Comments and Documentation
- **No inline comments**: Remove all `#` comments from function implementations
- **No docstrings in implementations**: Function implementations should not contain triple-quoted docstrings
- **Clean, minimal code**: Focus on clear, self-documenting code without explanatory comments

### Example Operator Implementation Checklist
- [ ] Create `src/beignet/_foo.py` with the operator implementation
- [ ] Add import and export in `src/beignet/__init__.py`
- [ ] Create `tests/beignet/test__foo.py` with comprehensive tests using Hypothesis
- [ ] Create `benchmarks/bench_foo.py` with time and memory benchmarks
- [ ] Add documentation with NumPy-style docstrings including examples
- [ ] Add operator to appropriate section in `docs/reference/operators/`
- [ ] Verify `torch.compile(foo, fullgraph=True)` works
- [ ] Test gradient computation with `torch.autograd.gradcheck`
- [ ] Ensure batch dimension handling is correct
- [ ] Run linting and formatting tools