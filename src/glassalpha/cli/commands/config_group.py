"""Config command group with subcommands: list, template, cheat.

Consolidates config-related commands under one namespace.
"""

import typer

from . import config_cheat_cmd, config_list_cmd, config_template_cmd

# Create config command group
config_app = typer.Typer(
    name="config",
    help="Configuration templates and helpers.",
    no_args_is_help=True,
)

# Register subcommands
config_app.command(name="list")(config_list_cmd)
config_app.command(name="template")(config_template_cmd)
config_app.command(name="cheat")(config_cheat_cmd)
