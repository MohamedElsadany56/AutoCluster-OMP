# Limitations

- The parser is pattern-based and is not a full C compiler frontend.
- Supported transformations are limited to controlled K-Means and Fuzzy C-Means C code patterns.
- Pointer aliasing and arbitrary side effects are not fully analyzed.
- Centroid-update transformations with local buffers are future work.
- Clang AST integration is future work.
- GPU, OpenACC, CUDA, and SIMD generation are future work.
