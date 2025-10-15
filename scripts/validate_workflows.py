#!/usr/bin/env python3
"""Validate GitHub Actions workflow files for syntax and structure."""

import sys
from pathlib import Path

import yaml


def validate_workflow_structure(workflow_file: Path) -> bool:
    """Validate that a workflow file has the required structure."""
    errors = []

    try:
        with open(workflow_file, encoding="utf-8") as f:
            content = yaml.safe_load(f)
    except Exception as e:
        errors.append(f"Error reading {workflow_file}: {e}")
        return False

    # Every workflow should have these top-level keys
    if "name" not in content:
        errors.append(f"Workflow {workflow_file.name} missing 'name' field")

    # Note: YAML parser interprets 'on:' as boolean True key (PyYAML quirk)
    if True not in content:
        errors.append(f"Workflow {workflow_file.name} missing 'on' field (parsed as True key)")

    if "jobs" not in content:
        errors.append(f"Workflow {workflow_file.name} missing 'jobs' field")

    # Every job should have runs-on and steps
    if "jobs" in content:
        for job_name, job_config in content["jobs"].items():
            if "runs-on" not in job_config:
                errors.append(f"Job '{job_name}' in {workflow_file.name} missing 'runs-on' field")
            if "steps" not in job_config:
                errors.append(f"Job '{job_name}' in {workflow_file.name} missing 'steps' field")
            elif not isinstance(job_config["steps"], list):
                errors.append(f"Job '{job_name}' steps should be a list")

    # Print all errors at once
    for error in errors:
        print(f"❌ {error}")

    return len(errors) == 0


def main():
    """Main validation function."""
    workflows_dir = Path(__file__).parent.parent / ".github" / "workflows"

    if not workflows_dir.exists():
        print("❌ No .github/workflows directory found")
        return 1

    all_valid = True

    for workflow_file in workflows_dir.glob("*.yml"):
        print(f"🔍 Validating {workflow_file.name}...")

        if not validate_workflow_structure(workflow_file):
            all_valid = False

    if all_valid:
        print("✅ All workflow files are valid")
        return 0
    else:
        print("❌ Some workflow files have issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
