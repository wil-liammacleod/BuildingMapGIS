# AI Agent Instructions

This document contains essential context, commands, and guidelines for AI agents working in this repository.

## Environment & Execution

- **Toolchain**: This project uses `uv` for Python dependency management and script execution.
- **Running Scripts**: You must **ALWAYS** use `uv run` to execute Python scripts or tools. This ensures the command runs in the correct isolated environment with all required dependencies.
  - *Standard Script*: `uv run file.py` (e.g., `uv run src/api_test.py`)
  - *Streamlit App*: `uv run streamlit run src/app.py`
  - Do not use the standard `python` command directly.

## Documentation & References

- **Whitebox Workflows**: The project utilizes Whitebox Workflows for advanced geospatial processing. A local copy of the Python API documentation is available here:
  - **Path**: `docs/Whitebox Workflows for Python User Manual.html`
  - Read this manual when you need to understand specific tool parameters, raster analysis functions, or data structures.

## General Best Practices

- **Read Before Writing**: Check `NOTES.md` or existing code in the `src/` directory to align with current project conventions before introducing new patterns.
- **Paths**: Treat the directory containing this file as the project root. Use paths relative to the root when reading files or executing commands.
- **Modifications**: Keep changes concise, modular, and directly aligned with the requested task. 
