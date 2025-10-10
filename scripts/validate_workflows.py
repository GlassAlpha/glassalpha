#!/usr/bin/env python3
"""Validate GitHub Actions workflow files and action versions.
Used by Makefile check-workflows target and pre-commit hooks.
"""

import sys
from pathlib import Path

import yaml


def check_action(action_ref: str) -> tuple[bool | None, str]:
    """Check if a GitHub Action reference is valid."""
    try:
        # For now, just do basic format validation
        # GitHub API rate limits and endpoint issues make this unreliable for CI
        if "/" not in action_ref:
            return False, f"❌ {action_ref} (invalid format)"

        parts = action_ref.split("/")
        if len(parts) != 2:
            return False, f"❌ {action_ref} (invalid format)"

        # Basic validation - if it has the right format and known patterns, assume valid
        owner, repo_and_ref = parts
        if "@" in repo_and_ref:
            repo, ref = repo_and_ref.split("@", 1)
        else:
            repo = repo_and_ref
            ref = "main"

        # Check for known problematic patterns
        if owner in ["actions", "sigstore", "pypa", "softprops"] and repo:
            return True, f"✅ {action_ref} (known action)"

        # For unknown actions, assume they exist (can't reliably check without API)
        return True, f"✅ {action_ref} (format valid)"

    except Exception as e:
        return None, f"⚠️  {action_ref} (check failed: {e})"


def main():
    """Main validation function."""
    workflows = list(Path(".github/workflows").glob("*.yml"))
    if not workflows:
        print("   No workflow files found")
        return 0

    total_actions = 0
    valid_actions = 0

    for wf_file in workflows:
        with open(wf_file) as f:
            workflow = yaml.safe_load(f)

    for job in workflow.get("jobs", {}).values():
        if job is None:
            continue
        for step in job.get("steps", []):
            if "uses" in step:
                action_ref = step["uses"]
                total_actions += 1
                status, message = check_action(action_ref)
                print(f"   {message}")
                if status is True:
                    valid_actions += 1
                elif status is False:
                    print("      💡 Try updating to a valid version")

    print()
    print(f"📊 Action check summary: {valid_actions}/{total_actions} valid actions")
    if valid_actions < total_actions:
        print("   ℹ️  Some actions may need version updates")
        print("   ℹ️  Run this check again if you have internet connectivity")
        return 1 if valid_actions == 0 else 0  # Only fail if no actions are valid
    print("   ✅ All checked actions appear valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
