"""Build the SVD cache for a model (run once per model, by one person, on the shared FS).

    python -m lorasub.cache_svd --model meta-llama/Llama-3.2-1B --svd_cache $SHARED/svd
"""
from __future__ import annotations

import argparse

from .modeling import load_model
from .models import target_modules_for
from .spectral import compute_svd_cache, list_cached


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--svd_cache", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    model = load_model(a.model, dtype="float32" if not __import__("torch").cuda.is_available() else "bfloat16")
    names = compute_svd_cache(model, a.svd_cache, a.model, suffixes=target_modules_for(a.model), overwrite=a.overwrite)
    print(f"cached {len(names)} modules; {len(list_cached(a.svd_cache, a.model))} files present")


if __name__ == "__main__":
    main()
