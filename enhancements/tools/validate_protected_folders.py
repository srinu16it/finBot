#!/usr/bin/env python3
"""
Validation script to ensure protected folders remain unmodified.

This script can be run in CI/CD pipelines or manually to verify
that no files in protected folders have been changed.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Set


PROTECTED_FOLDERS = [
    "myfirstfinbot/",
    # Add more protected folders here as needed
]


def get_modified_files() -> Set[str]:
    """
    Get list of modified files using git diff.
    
    Returns:
        Set of modified file paths
    """
    try:
        # Get modified files compared to main/master branch
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "origin/main"],
            capture_output=True,
            text=True,
            check=True
        )
        return set(result.stdout.strip().split('\n')) if result.stdout else set()
    except subprocess.CalledProcessError:
        # Try with master branch if main doesn't exist
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD", "origin/master"],
                capture_output=True,
                text=True,
                check=True
            )
            return set(result.stdout.strip().split('\n')) if result.stdout else set()
        except subprocess.CalledProcessError:
            print("Error: Could not get git diff. Make sure you're in a git repository.")
            sys.exit(1)


def check_protected_folders(modified_files: Set[str]) -> List[str]:
    """
    Check if any modified files are in protected folders.
    
    Args:
        modified_files: Set of modified file paths
        
    Returns:
        List of violations (files in protected folders that were modified)
    """
    violations = []
    
    for file_path in modified_files:
        for protected_folder in PROTECTED_FOLDERS:
            if file_path.startswith(protected_folder):
                violations.append(file_path)
                break
    
    return violations


def main():
    """Main function to run the validation."""
    print("Checking for modifications in protected folders...")
    
    # Get modified files
    modified_files = get_modified_files()
    
    if not modified_files:
        print("✅ No modified files detected.")
        return 0
    
    print(f"Found {len(modified_files)} modified files.")
    
    # Check for violations
    violations = check_protected_folders(modified_files)
    
    if violations:
        print("\n❌ ERROR: Modifications detected in protected folders!")
        print("\nThe following files in protected folders have been modified:")
        for violation in violations:
            print(f"  - {violation}")
        
        print(f"\nProtected folders that must not be modified:")
        for folder in PROTECTED_FOLDERS:
            print(f"  - {folder}")
        
        print("\nPlease revert these changes or move your modifications to the enhancements/ folder.")
        return 1
    
    print("✅ No modifications in protected folders. All good!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 