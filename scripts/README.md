
### validate_notebooks.py

Validates example notebooks for QA testing.

**Usage:**
```bash
# Validate all notebooks
python scripts/validate_notebooks.py

# Validate specific notebook
python scripts/validate_notebooks.py examples/notebooks/quickstart_from_model.ipynb

# Via Makefile
make check-notebooks
```

**Checks:**
- Notebook structure is valid JSON
- Contains code cells
- Has at least one GlassAlpha import (`import glassalpha` or `from glassalpha`)

**Exit codes:**
- `0`: All notebooks valid
- `1`: At least one notebook failed validation

**Example output:**
```
✅ quickstart_from_model.ipynb
   Code cells: 6
   GlassAlpha import: True
❌ broken_notebook.ipynb
   Code cells: 0
   GlassAlpha import: False
   ⚠️  No GlassAlpha import found
```
