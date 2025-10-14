"""Cache management commands."""

import typer

from glassalpha.report.cache import clear_cache, get_cache_dir

app = typer.Typer(help="Manage PDF export cache")


@app.command()
def clear():
    """Clear PDF export cache."""
    count = clear_cache()
    if count > 0:
        typer.echo(f"✓ Cleared {count} cached PDFs")
    else:
        typer.echo("Cache is already empty")


@app.command()
def info():
    """Show cache information."""
    cache_dir = get_cache_dir()
    pdfs = list(cache_dir.glob("*.pdf"))

    if pdfs:
        total_size = sum(p.stat().st_size for p in pdfs)
        typer.echo(f"Cache directory: {cache_dir}")
        typer.echo(f"Cached PDFs: {len(pdfs)}")
        typer.echo(f"Total size: {total_size:,} bytes ({total_size / 1024 / 1024:.1f} MB)")
    else:
        typer.echo("Cache is empty")
