"""Fetch MACE-MP-0 medium through mace-torch and record its identity in results/mace_model.json."""
from __future__ import annotations

import json
import urllib.request
from importlib import metadata

from dftgnn.graphs.store import sha256_file
from dftgnn.io.results import write_result
from dftgnn.mlip import mace_model as M


def main() -> None:
    from mace.calculators.foundations_models import mace_mp_urls

    p = M.model_path()
    with urllib.request.urlopen(M.LISTING_URL, timeout=30) as r:
        listing = json.loads(r.read())
    asset = next(a for a in listing["assets"] if a["name"] == M.FILE_NAME)
    pay = {"model": "MACE-MP-0 medium", "mace_torch_key": M.MODEL_KEY, "mace_torch_url": mace_mp_urls[M.MODEL_KEY],
           "release_url": M.RELEASE_URL, "release_listing_url": M.LISTING_URL,
           "release_tag": listing["tag_name"], "release_name": listing["name"],
           "listing_asset": {"name": asset["name"], "size": asset["size"],
                             "browser_download_url": asset["browser_download_url"],
                             "updated_at": asset.get("updated_at")},
           "file_name": p.name, "file_size_bytes": p.stat().st_size, "sha256": sha256_file(p),
           "size_matches_listing": p.stat().st_size == asset["size"],
           "mace_torch_version": metadata.version("mace-torch"), "dtype": "float64",
           "cache_path": str(p.relative_to(M.REPO_ROOT))}
    print(json.dumps(pay, indent=1))
    write_result("mace_model", pay)


if __name__ == "__main__":
    main()
