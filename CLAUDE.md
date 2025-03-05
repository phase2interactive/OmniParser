# OmniParser Development Guidelines

## Environment Setup
```bash
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

## Commands
- **Run Demo**: `python gradio_demo.py`
- **Run Tests**: `pytest -xvs tests/`
- **Run Single Test**: `pytest -xvs tests/test_file.py::test_function`
- **Lint Code**: `ruff check .`
- **Format Code**: `ruff format .`

## Code Style
- **Imports**: Group imports by standard lib, third-party, then local modules
- **Typing**: Use type hints for function parameters and return values
- **Naming**: 
  - Classes: `PascalCase`
  - Functions/variables: `snake_case` 
  - Constants: `UPPER_CASE`
- **Error Handling**: Use try/except blocks with specific exceptions
- **Documentation**: Include docstrings for classes and functions
- **Logging**: Use the built-in logging module instead of print statements

## Project Structure
- `util/`: Core parser utilities
- `omnitool/`: GUI agent tools and interfaces
- `eval/`: Evaluation scripts and benchmarks