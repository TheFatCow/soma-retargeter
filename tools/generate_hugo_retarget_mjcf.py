# SPDX-License-Identifier: Apache-2.0

"""Generate (or --check) configs/hugo/hugo_retarget.xml from hugo-mjlab's robot.mjcf.

    python tools/generate_hugo_retarget_mjcf.py            # write
    python tools/generate_hugo_retarget_mjcf.py --check    # exit 1 if stale

The output is hugo-mjlab's robot unchanged except for two massless, jointless
``*_hand_tcp`` bodies at the forearm tips (see soma_retargeter/assets/hugo.py).
Re-run after hugo-mjlab's MJCF changes; --check is how you notice that it did.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soma_retargeter.assets.hugo as hugo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="hugo-mjlab robot.mjcf (default: sibling checkout)")
    ap.add_argument("--output", default=str(hugo.DEFAULT_OUTPUT_PATH))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    xml = hugo.build_retarget_spec(args.source, out.parent).to_xml()
    if args.check:
        if not out.is_file() or out.read_text() != xml:
            print(f"[STALE] {out} does not match {hugo.resolve_source_mjcf_path(args.source)}; "
                  "re-run without --check.")
            return 1
        print(f"[OK] {out} is current.")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    print(f"[INFO] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
