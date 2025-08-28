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
- **Type promotion**: Use `torch.promote_types` for proper dtype handling across multiple input tensors, following PyTorch's type promotion rules

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
- **Descriptive names, no abbreviations**: Use full, descriptive variable names instead of abbreviations for clarity, including common abbreviations. Exception: PyTorch conventions like `out` parameter should be preserved
- **Avoid unnecessary suffixes/prefixes**: Remove redundant words like `_parameter`, `_value`, `_variable` when the context is clear
- **Mathematical variables**: Even single-letter mathematical variables should be lowercase (e.g., `n` not `N`, `r2` not `R2`)
- **Typographically pleasing names**: Variable names should be visually balanced and aesthetically pleasing. Prefer words with balanced letter shapes and avoid words that are visually top-heavy, bottom-heavy, or have awkward proportions:
  - **Good patterns**: Words with balanced mix of ascending/descending letters (g, j, p, q, y) and neutral letters, creating visual harmony
  - **Avoid**: Words dominated by tall letters (b, d, f, h, k, l, t) or short letters without visual interest
  - **Examples**:
    - ✅ Good: `maximum`, `minimum`, `beginning`, `current`, `degrees_of_freedom`, `sample_size`, `chi_squared`
    - ❌ Bad: `initial`, `first`, `final`, `height`, `length`, `width`, `critical_value`, `total`, `significance_level`, `confidence_interval`
- **Common abbreviations to avoid**:
  - `max` → `maximum`
  - `min` → `minimum` 
  - `std` → `standard_deviation` (except in function names like `torch.std`)
  - `sqrt` → `square_root`
  - `init` → avoid (use `starting`, `beginning` instead)
  - `curr` → `current`
  - `prev` → `previous`
  - `temp` → descriptive name for what it temporarily holds
  - `num` → `number`
- **Examples**:
  - ✅ `sample_size`, `effect_size`, `degrees_of_freedom`, `noncentrality`, `standard_error`, `maximum_iterations`, `square_root_two`
  - ❌ `sampleSize`, `N`, `R2`, `ncp`, `se`, `df`, `max_iterations`, `sqrt_two`, `temp_value`

### Statement Spacing
- **Add newlines between statements of different line lengths**: When consecutive statements have different character counts, separate them with blank lines for improved readability
- **Apply to all statement types**: This rule applies to assignments, control flow, function calls, and any other statements
- **Examples**:
  ```python
  # Correct - newlines between different length statements
  sample_size_tensor = torch.as_tensor(sample_size)

  groups_tensor = torch.as_tensor(groups)

  covariate_r2_tensor = torch.as_tensor(covariate_r2)

  num_covariates_tensor = torch.as_tensor(n_covariates)
  
  # Same length statements can be grouped together
  effect_size = effect_size.to(dtype)
  sample_size = sample_size.to(dtype)
  
  # Control flow also follows this rule
  max_iterations = 24

  for _ in range(max_iterations):
      # loop body
  
  result = torch.clamp(effect_size, min=0.0)

  if scalar_output:
      # if body
  ```

### Magic Numbers and Constants
- **Name magic numbers**: Replace unnamed numeric literals with descriptively named variables
- **Examples**:
  ```python
  # Correct - named constants
  minimum_effect_size = 1e-8
  maximum_effect_size_epsilon = 1e-6
  max_expansion_iterations = 8
  max_bisection_iterations = 24
  
  # Incorrect - magic numbers
  effect_size_lower = torch.zeros_like(initial_effect_size) + 1e-8
  for _ in range(8):
  ```

### Expression Inlining vs Breaking Out
- **Inline simple expressions that fit within line length limits**: If the entire statement fits reasonably on one line (under ~79 characters), inline the expression
- **Break out complex or multi-line expressions**: Split expressions that would span multiple lines or are conceptually complex
- **Examples**:
  ```python
  # Correct - simple expression inlined
  output = torch.clamp((1 - torch.erf(z_score / sqrt_2)) / 2, 0.0, 1.0)
  
  # Incorrect - unnecessary intermediate variable for simple expression
  power = (1 - torch.erf(z_score / sqrt_2)) / 2
  output = torch.clamp(power, 0.0, 1.0)
  
  # Correct - complex expression broken down with intermediate variables
  sqrt_df_over_n = torch.sqrt(degrees_of_freedom_1 / torch.clamp(sample_size_clamped, min=1.0))
  sqrt_residual_variance = torch.sqrt(torch.clamp(1.0 - covariate_r2_clamped, min=torch.finfo(dtype).eps))
  initial_effect_size = torch.clamp(
      (z_alpha + z_beta) * sqrt_df_over_n * sqrt_residual_variance,
      min=1e-8,
  )
  
  # Incorrect - complex nested expression that's hard to read
  effect_size_f0 = torch.clamp(
      (z_alpha + z_beta)
      * torch.sqrt(df1 / torch.clamp(n, min=1.0))
      * torch.sqrt(torch.clamp(1.0 - r2, min=torch.finfo(dtype).eps)),
      min=1e-8,
  )
  ```

### Temporary Variables
- **Eliminate cryptic temporary variables**: Don't use `var0`, `temp`, `x`, etc. Use descriptive names throughout the transformation process
- **Examples**:
  ```python
  # Correct - descriptive temporary variables
  sample_size_tensor = torch.as_tensor(sample_size)
  sample_size_1d = torch.atleast_1d(sample_size_tensor)
  sample_size_clamped = torch.clamp(sample_size_1d.to(dtype), min=3.0)
  
  # Incorrect - cryptic temporary variables
  n0 = torch.as_tensor(sample_size)
  n = torch.atleast_1d(n0)
  n = torch.clamp(n.to(dtype), min=3.0)
  ```

### Boolean Variable Naming
- **Use predicate names**: Boolean variables should clearly indicate true/false conditions without unnecessary prefixes
- **Examples**:
  ```python
  # Correct - clear predicates
  needs_expansion = power_high < power
  power_too_low = power_mid < power
  scalar_output = tensor.ndim == 0
  converged = torch.abs(power_diff) < tolerance
  
  # Incorrect - unclear or verbose
  flag = power_high < power
  check = power_mid < power
  is_scalar = tensor.ndim == 0
  condition = power_mid < target_power
  ```

### Tensor Transformation Chains
- **Name each transformation step**: Each step in tensor processing should have a descriptive name showing the transformation
- **Don't reuse variable names**: Avoid obscuring the transformation pipeline by reusing the same variable name
- **Examples**:
  ```python
  # Correct - transformation pipeline is clear
  sample_size_tensor = torch.as_tensor(sample_size)
  sample_size_1d = torch.atleast_1d(sample_size_tensor)
  sample_size_clamped = torch.clamp(sample_size_1d.to(dtype), min=3.0)
  
  # Incorrect - reusing variable names obscures the pipeline
  sample_size = torch.as_tensor(sample_size)
  sample_size = torch.atleast_1d(sample_size)
  sample_size = torch.clamp(sample_size.to(dtype), min=3.0)
  ```

### Mathematical Formula Variables
- **Break formulas into named components**: Complex formulas should be split into intermediate variables that reflect mathematical meaning
- **Examples**:
  ```python
  # Correct - mathematical components are clear
  degrees_of_freedom_1 = torch.clamp(groups_clamped - 1.0, min=1.0)
  sqrt_df_over_n = torch.sqrt(degrees_of_freedom_1 / torch.clamp(sample_size_clamped, min=1.0))
  sqrt_residual_variance = torch.sqrt(torch.clamp(1.0 - covariate_r2_clamped, min=torch.finfo(dtype).eps))
  initial_effect_size = (z_alpha + z_beta) * sqrt_df_over_n * sqrt_residual_variance
  
  # Incorrect - formula is opaque
  result = (z_alpha + z_beta) * torch.sqrt(df1 / torch.clamp(n, min=1.0)) * torch.sqrt(torch.clamp(1.0 - r2, min=eps))
  ```

### Algorithm Phase Naming
- **Variables should indicate algorithm phases**: Names should show which phase of the algorithm they belong to
- **Examples**:
  ```python
  # Correct - algorithm phases are clear
  initial_effect_size = calculate_initial_estimate(...)
  effect_size_lower = torch.zeros_like(initial_effect_size) + minimum_effect_size
  power_high = analysis_of_covariance_power(effect_size_upper, ...)
  effect_size_mid = (effect_size_lower + effect_size_upper) * 0.5
  
  # Incorrect - no indication of algorithm structure
  val1 = calculate_initial(...)
  val2 = torch.zeros_like(val1) + min_val
  val3 = test_function(upper_val, ...)
  val4 = (lower + upper) * 0.5
  ```

### Iteration and Loop Variables
- **Use descriptive names when context matters**: Loop variables should be descriptive when the iteration has specific meaning
- **Examples**:
  ```python
  # Correct - when iterations have specific algorithmic meaning
  for expansion_step in range(max_expansion_iterations):
  for bisection_step in range(max_bisection_iterations):
  
  # Acceptable - when it's truly just counting
  for _ in range(max_iterations):
  
  # Incorrect - generic names when context matters
  for i in range(max_expansion_iterations):
  for j in range(max_bisection_iterations):
  ```

### Return Value Preparation
- **Final processing should be obvious**: Variables for output preparation should clearly indicate their purpose
- **Examples**:
  ```python
  # Correct - output preparation is obvious
  result = torch.clamp(effect_size_mid, min=0.0)
  result_scalar = result.reshape(())
  
  # Incorrect - unclear what these represent
  output = torch.clamp(x, min=0.0)
  out = output.reshape(())
  temp = result.reshape(())
  ```

### Typography and Visual Balance
- **Consistent indentation alignment**: Align continuation lines and multi-line expressions for visual clarity
- **Balanced parentheses and brackets**: When breaking expressions across lines, align opening and closing delimiters
- **Symmetric spacing around operators**: Maintain consistent spacing around mathematical and logical operators
- **Visual grouping through whitespace**: Use blank lines to create visual separation between logical code blocks
- **Examples**:
  ```python
  # Correct - well-aligned multi-line expressions
  initial_effect_size = torch.clamp(
      (z_alpha + z_beta) * sqrt_df_over_n * sqrt_residual_variance,
      min=minimum_effect_size,
  )
  
  scalar_output = (
      sample_size_tensor.ndim == 0
      and groups_tensor.ndim == 0
      and covariate_r2_tensor.ndim == 0
      and num_covariates_tensor.ndim == 0
  )
  
  # Correct - consistent spacing around operators
  power_too_low = power_mid < target_power
  effect_size_mid = (effect_size_lower + effect_size_upper) * 0.5
  adjustment_factor = (degrees_of_freedom + 2.0) / torch.clamp(degrees_of_freedom, min=1.0)
  
  # Incorrect - poor alignment and inconsistent spacing
  initial_effect_size = torch.clamp(
  (z_alpha + z_beta) * sqrt_df_over_n * sqrt_residual_variance,
  min=minimum_effect_size,
  )
  
  scalar_output = (sample_size_tensor.ndim == 0 and
                   groups_tensor.ndim == 0 and
                   covariate_r2_tensor.ndim == 0)
  
  power_too_low=power_mid<target_power
  effect_size_mid=(effect_size_lower+effect_size_upper)*0.5
  ```

### Function Call Formatting
- **Break long function calls consistently**: When function calls exceed line length, break at logical parameter boundaries
- **Align parameters vertically**: Parameters should align for readability when broken across lines
- **Trailing commas on multi-line calls**: Use trailing commas for multi-line function calls to minimize diff noise
- **Examples**:
  ```python
  # Correct - clean parameter alignment
  power_mid = analysis_of_covariance_power(
      effect_size_mid,
      sample_size_clamped,
      groups_clamped,
      covariate_r2_clamped,
      num_covariates_clamped,
      alpha,
  )
  
  # Correct - short calls can stay inline
  result = torch.clamp(effect_size_mid, min=0.0)
  
  # Incorrect - poor alignment
  power_mid = analysis_of_covariance_power(effect_size_mid,
                                         sample_size_clamped,
                                         groups_clamped,
                                         covariate_r2_clamped,
                                         num_covariates_clamped,
                                         alpha)
  ```

### Line Length and Visual Density
- **Target line length around 79-88 characters**: Aim for readability without excessive wrapping
- **Avoid overly dense lines**: Break up lines that pack too much information
- **Balance vertical and horizontal space**: Use both line breaks and spacing to create readable code
- **Examples**:
  ```python
  # Correct - appropriate information density
  covariate_r2_clamped = torch.clamp(
      covariate_r2_1d.to(dtype), min=0.0, max=1 - torch.finfo(dtype).eps
  )
  
  maximum_expansion_iterations = 8
  
  for _ in range(maximum_expansion_iterations):
      # loop body with appropriate spacing
  
  # Incorrect - too dense, hard to parse
  covariate_r2_clamped = torch.clamp(covariate_r2_1d.to(dtype), min=0.0, max=1 - torch.finfo(dtype).eps)
  maximum_expansion_iterations = 8
  for _ in range(maximum_expansion_iterations):
  ```

### Comments and Documentation
- **No inline comments**: Remove all `#` comments from function implementations
- **No docstrings in implementations**: Function implementations should not contain triple-quoted docstrings
- **Clean, minimal code**: Focus on clear, self-documenting code without explanatory comments

### Type Promotion
- **Use torch.promote_types**: Always use `torch.promote_types` for dtype determination across multiple input tensors
- **Start with float32**: Begin type promotion from `torch.float32` as the base dtype for statistical computations
- **Avoid manual dtype checking**: Do not use manual `if tensor.dtype == torch.float64` patterns
- **Examples**:
  ```python
  # Correct - using torch.promote_types
  dtype = torch.float32
  for tensor in (effect_size, sample_size, groups):
      dtype = torch.promote_types(dtype, tensor.dtype)
  
  # Incorrect - manual dtype checking
  if effect_size.dtype == torch.float64 or sample_size.dtype == torch.float64:
      dtype = torch.float64
  else:
      dtype = torch.float32
  ```

### Beautiful Code Principles
Functions like `_glass_delta` exemplify genuinely beautiful code through these principles:

- **Perfect Minimalism**: Use simple `torch.promote_types(input.dtype, other.dtype)` for two tensors instead of `functools.reduce` when unnecessary. Import only what's essential.
- **Natural Visual Rhythm**: Group related operations with logical blank lines. Create natural blocks for input processing, dtype handling, conversions, and calculations.
- **Elegant Variable Reuse**: Transform variables in place when it creates cleaner code:
  ```python
  output = torch.clamp(other_variance, min=torch.finfo(dtype).eps)
  output = (input_mean - other_mean) / torch.sqrt(output)
  ```
- **Mathematical Elegance**: Write direct computations without overly verbose statistical terminology. Use essential safety measures (like single strategic clamps) only where truly needed.
- **Perfect Proportions**: Maintain balanced line lengths and natural flow from simple to complex operations. Let the mathematics speak through clean, unforced structure.
- **Principle of Least Complexity**: Follow the path of natural beauty - don't try to be fancy, just be elegant through simplicity and logical clarity.

**Examples of beautiful patterns**:
```python
# Beautiful - natural grouping and flow
input = torch.atleast_1d(input)
other = torch.atleast_1d(other)

dtype = torch.promote_types(input.dtype, other.dtype)

input = input.to(dtype)
other = other.to(dtype)

input_mean = torch.mean(input, dim=-1)
other_mean = torch.mean(other, dim=-1)

# Beautiful - elegant variable transformation
output = torch.clamp(variance, min=torch.finfo(dtype).eps)  
output = (mean_diff) / torch.sqrt(output)

# Avoid - artificial complexity
effect_size_numerator_component = torch.mean(input_tensor_values, dim=-1)
effect_size_denominator_component = torch.mean(other_tensor_values, dim=-1)  
standardized_effect_size_computation = compute_effect_size_with_pooled_variance(...)
```

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