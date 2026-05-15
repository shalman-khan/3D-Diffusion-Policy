#!/bin/bash
# Patches accelerate's torch_xla.py which imports pkg_resources unconditionally,
# breaking on environments where pkg_resources is not available.
# Safe to run multiple times.

TORCH_XLA=$(python3 -c "
import accelerate, os
print(os.path.join(os.path.dirname(accelerate.__file__), 'utils', 'torch_xla.py'))
")

if [ ! -f "$TORCH_XLA" ]; then
    echo "[fix_env] torch_xla.py not found — skipping"
    exit 0
fi

if grep -q "except ImportError" "$TORCH_XLA"; then
    echo "[fix_env] Already patched: $TORCH_XLA"
else
    sed -i 's/^import pkg_resources$/try:\n    import pkg_resources\nexcept ImportError:\n    pkg_resources = None/' "$TORCH_XLA"
    echo "[fix_env] Patched: $TORCH_XLA"
fi
