# Protected Folders Policy

## Overview

Certain folders in the FinBot repository are designated as **protected** and must not be modified under any circumstances. This policy ensures the integrity of legacy code and maintains a clear separation between original functionality and enhancements.

## Protected Folders

The following folders and all their contents are protected:

### 1. `myfirstfinbot/`
- **Status**: PROTECTED - DO NOT MODIFY
- **Reason**: Contains legacy code that must remain untouched
- **Scope**: All files and subdirectories within this folder

## Policy Enforcement

### 1. **Pre-commit Hooks**
Pre-commit hooks are configured to skip protected folders:
- Black formatter will not touch these files
- Ruff linter will not check these files
- File fixers (trailing whitespace, EOF) will skip these folders

### 2. **CI/CD Validation**
A validation script runs in CI/CD to ensure no protected files are modified:
```bash
make validate
# or
python enhancements/tools/validate_protected_folders.py
```

### 3. **Code Review**
All pull requests are reviewed to ensure compliance with this policy.

## What to Do Instead

If you need to extend or modify functionality that exists in protected folders:

### 1. **Create a Wrapper**
```python
# enhancements/wrappers/enhanced_feature.py
from myfirstfinbot.some_module import OriginalClass

class EnhancedClass(OriginalClass):
    """Enhanced version of OriginalClass with new features."""
    
    def new_method(self):
        """New functionality added without modifying original."""
        pass
```

### 2. **Use Composition**
```python
# enhancements/extensions/composed_feature.py
from myfirstfinbot.some_module import OriginalClass

class ComposedFeature:
    """Composed class that uses OriginalClass internally."""
    
    def __init__(self):
        self._original = OriginalClass()
    
    def enhanced_operation(self):
        """Enhanced operation using the original class."""
        result = self._original.some_method()
        # Add enhancements here
        return enhanced_result
```

### 3. **Create New Implementations**
If you need completely different behavior, create new implementations in the `enhancements/` folder rather than modifying protected code.

## Validation

To check if you've accidentally modified protected folders:

```bash
# Run the validation script
make validate

# Or manually check with git
git diff --name-only | grep "^myfirstfinbot/"
```

## Exceptions

There are **NO EXCEPTIONS** to this policy. If you believe a change to a protected folder is absolutely necessary:

1. Document the requirement thoroughly
2. Discuss with the team lead
3. Consider alternative approaches
4. If approved, the change must go through a special review process

## Consequences

Violations of this policy will result in:
1. Pull request rejection
2. Required reversion of changes
3. Additional code review requirements

## Questions?

If you're unsure whether a change violates this policy, ask before making the change. It's better to clarify beforehand than to have to revert work later. 