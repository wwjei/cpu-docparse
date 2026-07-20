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

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
mkdir -p "$MODELS_DIR"
export MODELS_DIR

echo "==> Preparing PP-DocLayoutV3 layout model"

if [[ -f "$MODELS_DIR/PP-DocLayoutV3.onnx" ]]; then
  echo "    already exists: $MODELS_DIR/PP-DocLayoutV3.onnx — skipping"
else
  python - <<'PY'
import os, subprocess, sys, glob

models_dir = os.environ.get("MODELS_DIR", "models")
pd_dir = os.path.join(models_dir, "PP-DocLayoutV3")

# 1) Download official Paddle inference model
if not os.path.isdir(pd_dir) or not glob.glob(os.path.join(pd_dir, "**/*.pdmodel"), recursive=True):
    print("    downloading official Paddle inference model ...")
    # 方式一: paddlex CLI (兼容各版本)
    ret = subprocess.run(
        [sys.executable, "-m", "paddlex", "--download_model", "PP-DocLayoutV3",
         "--save_dir", pd_dir],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        # 方式二: 旧版 paddlex API
        try:
            from paddlex.inference.utils.official_models import OfficialModelsResolver
            resolver = OfficialModelsResolver()
            resolved = resolver.resolve("PP-DocLayoutV3")
            pd_dir = resolved
        except ImportError:
            pass
        # 方式三: 直接 URL 下载
        if not glob.glob(os.path.join(pd_dir, "**/*.pdmodel"), recursive=True):
            print("    paddlex CLI failed, trying direct download ...")
            import urllib.request, zipfile, io
            url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayoutV3_infer.tar"
            print(f"    downloading from {url}")
            tar_path = os.path.join(models_dir, "PP-DocLayoutV3_infer.tar")
            urllib.request.urlretrieve(url, tar_path)
            import tarfile
            with tarfile.open(tar_path) as tf:
                tf.extractall(models_dir)
            os.remove(tar_path)
            # 解压后目录可能是 PP-DocLayoutV3_infer
            for candidate in ["PP-DocLayoutV3_infer", "PP-DocLayoutV3"]:
                if os.path.isdir(os.path.join(models_dir, candidate)):
                    pd_dir = os.path.join(models_dir, candidate)
                    break
    print(f"    model dir: {pd_dir}")

# Locate the model definition file (.pdmodel or .json for Paddle 3.0 PIR format)
pdmodel = None
for root, _, files in os.walk(pd_dir):
    for f in files:
        if f.endswith(".pdmodel"):
            pdmodel = os.path.join(root, f)
            break
    if pdmodel:
        break
    for f in files:
        if f == "inference.json":
            pdmodel = os.path.join(root, f)
            break
    if pdmodel:
        break
if pdmodel is None:
    sys.exit("    ERROR: could not locate .pdmodel or inference.json under " + pd_dir + "\n"
             "    请手动下载: https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayoutV3_infer.tar\n"
             "    解压到 models/ 目录后重新运行本脚本")
if pdmodel.endswith(".pdmodel"):
    pdiparams = pdmodel[:-len(".pdmodel")] + ".pdiparams"
else:
    pdiparams = os.path.join(os.path.dirname(pdmodel), "inference.pdiparams")

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
