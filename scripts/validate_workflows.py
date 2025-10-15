#!/usr/bin/env python3
"""Validate GitHub Actions workflow files for syntax and structure."""

import sys
import yaml
from pathlib import Path


def validate_workflow_structure(workflow_file: Path) -> bool:
    """Validate that a workflow file has the required structure."""
    try:
        with open(workflow_file, encoding="utf-8") as f:
            content = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error reading {workflow_file}: {e}")
        return False

    # Every workflow should have these top-level keys
    if "name" not in content:
        print(f"❌ Workflow {workflow_file.name} missing 'name' field")
        return False

    # Note: YAML parser interprets 'on:' as boolean True key (PyYAML quirk)
    if True not in content:
        print(f"❌ Workflow {workflow_file.name} missing 'on' field (parsed as True key)")
        return False

    if "jobs" not in content:
        print(f"❌ Workflow {workflow_file.name} missing 'jobs' field")
        return False

    # Every job should have runs-on and steps
    for job_name, job_config in content["jobs"].items():
        if "runs-on" not in job_config:
            print(f"❌ Job '{job_name}' in {workflow_file.name} missing 'runs-on' field")
            return False
        if "steps" not in job_config:
            print(f"❌ Job '{job_name}' in {workflow_file.name} missing 'steps' field")
            return False
        if not isinstance(job_config["steps"], list):
            print(f"❌ Job '{job_name}' steps should be a list")
            return False

    return True


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
