# Contributing to ASR ROVER

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/ASR-Mixture_of_expert-ROVER.git
cd ASR-Mixture_of_expert-ROVER
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Code Style

We follow PEP 8 guidelines with some modifications:

- Line length: 100 characters (soft limit)
- Use type hints where applicable
- Docstrings: Google style format
- Format code with `black`:
```bash
black src/ examples/ tests/
```

## Testing

Run tests before submitting PR:
```bash
pytest tests/
```

## Submitting Changes

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Add: description of your changes"
```

3. Push to your fork:
```bash
git push origin feature/your-feature-name
```

4. Open a Pull Request on GitHub

## Commit Message Guidelines

Use conventional commits format:
- `Add:` new feature
- `Fix:` bug fix
- `Update:` update existing functionality
- `Refactor:` code refactoring
- `Docs:` documentation changes
- `Test:` test additions/changes

## Areas for Contribution

- Additional ASR system integrations
- Performance optimizations
- Documentation improvements
- Bug fixes
- Test coverage
- New output formats
- Language support improvements

## Questions?

Open an issue for discussion before starting major changes.
