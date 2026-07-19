#!/usr/bin/env bash
#
# Download and prepare model files for the CPU doc-parse pipeline.
#
# The layout model (PP-DocLayoutV3.onnx, ~125MB) is excluded from the git repo
# because it exceeds GitHub's 100MB file limit. This script fetches the official
# Paddle inference model and converts it to ONNX.
#
# The small SLANet_fixed.onnx (table structure) is already committed in models/.
#
# Requirements: paddlepaddle, paddle2onnx, paddlex  (pip install paddlepaddle paddle2onnx paddlex)
#
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models"
mkdir -p "$MODELS_DIR"

echo "==> Preparing PP-DocLayoutV3 layout model"

if [[ -f "$MODELS_DIR/PP-DocLayoutV3.onnx" ]]; then
  echo "    already exists: $MODELS_DIR/PP-DocLayoutV3.onnx — skipping"
else
  python - <<'PY'
import os, subprocess, sys

models_dir = os.environ.get("MODELS_DIR", "models")
pd_dir = os.path.join(models_dir, "PP-DocLayoutV3")

# 1) Download official Paddle inference model via PaddleX model registry
if not os.path.isdir(pd_dir):
    print("    downloading official Paddle inference model ...")
    from paddlex.inference.utils.official_models import OfficialModelsResolver
    resolver = OfficialModelsResolver()
    resolved = resolver.resolve("PP-DocLayoutV3")
    # resolved is a local cache dir containing inference.pdmodel/.pdiparams + inference.yml
    pd_dir = resolved
    print(f"    resolved cache dir: {pd_dir}")

# Locate the .pdmodel / .pdiparams files
pdmodel = None
for root, _, files in os.walk(pd_dir):
    for f in files:
        if f.endswith(".pdmodel"):
            pdmodel = os.path.join(root, f)
    if pdmodel:
        break
if pdmodel is None:
    sys.exit("    ERROR: could not locate inference.pdmodel under " + pd_dir)
pdiparams = pdmodel[:-len(".pdmodel")] + ".pdiparams"

out_onnx = os.path.join(models_dir, "PP-DocLayoutV3.onnx")
print(f"    converting {pdmodel} -> {out_onnx}")
subprocess.run([
    "paddle2onnx",
    "--model_dir", os.path.dirname(pdmodel),
    "--model_filename", os.path.basename(pdmodel),
    "--params_filename", os.path.basename(pdiparams),
    "--save_file", out_onnx,
    "--opset_version", "16",
    "--enable_onnx_checker", "True",
], check=True)
print("    done:", out_onnx)
PY
fi

echo "==> Verifying model files"
for f in PP-DocLayoutV3.onnx SLANet_fixed.onnx; do
  if [[ -f "$MODELS_DIR/$f" ]]; then
    printf "    [OK] %-28s %s\n" "$f" "$(du -h "$MODELS_DIR/$f" | cut -f1)"
  else
    printf "    [MISSING] %s\n" "$f"
  fi
done

echo "==> Done."
