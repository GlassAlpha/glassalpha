# Development Resources

This directory contains development artifacts, internal documentation, and testing resources that are not part of the main user-facing package.

## Directory Structure

### `demos/`
Demo scripts and examples used during development:
- `demo_foundation*.py` - Core architecture demonstrations
- `demo_integration.py` - Component integration examples
- `demo_summary.py` - Development summary scripts
- `demo_shap_plot.png` - Sample SHAP visualization output

### `ci/`
Continuous Integration and development infrastructure:
- `CI_*.md` - CI/CD research and solutions
- `CI_DIAGNOSIS.py` - CI troubleshooting scripts
- `COVERAGE_PLAN.md` - Test coverage strategy

### `reviews/`
Project reviews and status documentation:
- `HANDOFF.md` - Phase handoff documentation
- `ML_COMPONENTS_STATUS.md` - Component implementation status
- `PHASE2_PROMPT.md` - Phase 2 implementation guide
- `*_review.py` - Automated review scripts

### `tests/`
Development testing artifacts:
- `test_*.py` - Individual component tests
- `test_pdfs/` - Generated test PDF outputs
- `test_reports/` - Generated test HTML reports
- `htmlcov/` - Test coverage reports
- Various test audit PDFs

## Purpose

These files support development and internal processes but are not part of the production package. They have been moved here to:

1. **Clean main directory** - Keep the main package directory focused on user-facing content
2. **Preserve development history** - Maintain development artifacts for reference
3. **Organize by function** - Group related development resources together
4. **Improve professional presentation** - Remove clutter from the main repository view

## Usage

These files are primarily for:
- **Developers** working on GlassAlpha enhancements
- **Contributors** understanding the development process
- **Maintainers** reviewing project history and status
- **QA/Testing** validating functionality across different scenarios

## Note

Files in this directory are not installed with the main package and do not affect end-user functionality. They serve as development documentation and testing resources.
