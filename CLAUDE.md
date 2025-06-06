# Claude Code Development Notes

This file contains useful information discovered during development with Claude Code.

## Testing

### Running Tests
- Use `uv run pytest` to run tests in this project
- For verbose output: `uv run pytest -v`
- For hypothesis statistics: `uv run pytest --hypothesis-show-statistics`

### Hypothesis Testing
- When refactoring hypothesis tests, be careful with data type compatibility
- For scipy sparse matrices, ensure indices are `int32` to avoid buffer dtype mismatches:
  ```python
  # Ensure indices are int32 for scipy compatibility
  graph.indices = graph.indices.astype(numpy.int32)
  graph.indptr = graph.indptr.astype(numpy.int32)
  ```

### Performance Optimization
- For graph algorithm tests, use small graphs (2-5 nodes) instead of large ones for faster execution
- Remove expensive gradient checking (`torch.autograd.gradcheck`) for algorithms with discontinuities like shortest path
- Keep basic gradient flow tests but simplify them

## Common Issues

### Scipy CSR Matrix Compatibility
- scipy.sparse.csgraph functions expect `int32` indices, not `int64` 
- Always cast indices when creating sparse matrices for scipy compatibility:
  ```python
  numpy.array(row_indices, dtype=numpy.int32)
  numpy.array(col_indices, dtype=numpy.int32)
  ```

### PyTorch Sparse Tensors
- PyTorch sparse CSR tensor support is in beta (expect warnings)
- Use `.float()` to ensure consistent dtypes between scipy and torch results

## Implementing New Operators

### Graph Algorithms Implemented

#### Shortest Path Algorithms
- **Dijkstra**: Single-source, non-negative weights only
- **Bellman-Ford**: Single-source, handles negative weights, detects negative cycles
- **Floyd-Warshall**: All-pairs, handles negative weights, O(V³) complexity
- **Johnson**: All-pairs, uses Bellman-Ford + Dijkstra for sparse graphs

#### Graph Traversal
- **Breadth-First Search**: Level-by-level traversal
- **Depth-First Search**: Deep exploration with backtracking

#### Minimum Spanning Tree
- **Kruskal's Algorithm**: Union-Find based MST construction

#### Maximum Flow
- **Edmonds-Karp**: BFS-based Ford-Fulkerson implementation
- **Dinic's Algorithm**: Level graph and blocking flow approach

#### Graph Components
- **Strongly Connected Components**: Tarjan's algorithm for directed graphs
- **Weak Connected Components**: Union-Find for undirected connectivity

#### Other Algorithms
- **Hopcroft-Karp**: Maximum bipartite matching
- **Yen's Algorithm**: K shortest paths (simplified implementation)

### Gradient Support Considerations
- Avoid complex sparse tensor operations in gradient paths (PyTorch sparse support is limited)
- For Johnson's algorithm, fallback to Floyd-Warshall for small graphs to avoid gradient issues
- Use `.detach()` or simplified implementations when gradient support is problematic

### Algorithm Implementation Pattern
1. Create `_algorithm_name.py` file in `src/beignet/`
2. Add import to `__init__.py` in alphabetical order
3. Implement with proper docstring following NumPy style
4. Create/update test file with same pattern as `test__bellman_ford.py`
5. Use small graph sizes in tests for faster execution

## Development Workflow
1. Read existing code to understand patterns and conventions
2. Use TodoWrite tool to plan complex refactoring tasks
3. Test changes immediately after implementation
4. Check for both functionality and performance improvements
5. Run all related tests before considering implementation complete