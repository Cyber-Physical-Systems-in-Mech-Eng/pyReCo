#!/usr/bin/env python3
import sys
import re
import subprocess

# Get the diff of staged changes
diff_output = subprocess.run(
    ["git", "diff", "--cached", "--unified=0"], capture_output=True, text=True
).stdout

# Find added function definitions
added_functions = re.findall(r"^\+def (\w+)\(", diff_output, re.MULTILINE)

# Check if they have docstrings
errors = []
for func in added_functions:
    pattern = rf"^\+def {func}\(.*\):\n(\+\s+\"\"\"|\+\s+#)"
    if not re.search(pattern, diff_output, re.MULTILINE):
        errors.append(f"Function `{func}` is missing a docstring!")

# Report errors
if errors:
    print("\n".join(errors))
    sys.exit(1)
