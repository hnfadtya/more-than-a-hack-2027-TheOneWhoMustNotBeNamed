"""Download the released model weights and decoder settings into outputs/ and check their SHA-256.

    python -m mtah.get_weights              # release v1.0 of this repository
    python -m mtah.get_weights --tag v1.0 --force

outputs/ is not committed, so the files the final submission needs are GitHub Release assets.
Each asset has a flat, unique name in the release and is written to the path the pipeline reads.
"""
import argparse
import hashlib
import os
import sys
import urllib.request

from .common import ROOT

REPO = "hnfadtya/more-than-a-hack-2027-TheOneWhoMustNotBeNamed"

# release asset -> (path under the repository root, sha256)
ASSETS = {
    "run2_best.pt": ("outputs/nn/run2/best.pt",
                     "e21cba8b90ed56710395ba14253ac33d2556f9530aba44899e3dc8c523f1c684"),
    "run3_best.pt": ("outputs/nn/run3/best.pt",
                     "169ce493c27aa2be45c38b6e96974bc4edb5be2eb437b115c945eb1d4eeb0ad3"),
    "run5_best.pt": ("outputs/nn/run5/best.pt",
                     "31c75e5776afca1ebc9565f15ddae788f86456515c0632841b18a4439c7cbcff"),
    "run6_best.pt": ("outputs/nn/run6/best.pt",
                     "5ed3c9c544d4de079ac67a52f6cf457e339458a218cbf57f0f4fdea6f8a5b14a"),
    "ens235_decoder.json": ("outputs/nn/ens235/decoder.json",
                            "44e9db1b91d5921cf6a68fd3bbc3800e3703a626700e0c466ec930d04bad3f68"),
    "vball_ft1_best.pt": ("outputs/vball/ft1/best.pt",
                          "a07e9c5c1c53996877dd5d2ab76e735bc2e5c643c97c3a1020528c813bd5698f"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v1.0", help="release tag")
    ap.add_argument("--force", action="store_true", help="download even if a matching file exists")
    args = ap.parse_args()

    bad = 0
    for asset, (rel, want) in ASSETS.items():
        dst = os.path.join(ROOT, rel)
        if os.path.isfile(dst) and not args.force and sha256(dst) == want:
            print(f"  ok (already there)  {rel}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        url = f"https://github.com/{REPO}/releases/download/{args.tag}/{asset}"
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dst + ".part")
        got = sha256(dst + ".part")
        if got != want:
            os.remove(dst + ".part")
            print(f"  SHA-256 MISMATCH for {asset}: {got}")
            bad += 1
            continue
        os.replace(dst + ".part", dst)
        print(f"  ok  {rel}")
    if bad:
        return f"{bad} file(s) failed the SHA-256 check"
    print("all weights in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
