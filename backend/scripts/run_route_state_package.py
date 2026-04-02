from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.research_logic import (  # noqa: E402
    build_route_state_package_summary,
    compile_route_state_package,
    load_route_state_package_manifest,
    write_route_state_package_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Compile a reusable route-state package from packetized traces and optional L1 snapshots.'
    )
    parser.add_argument('--manifest', required=True, help='Path to a RouteStatePackageManifest JSON file.')
    parser.add_argument('--output-dir', required=True, help='Directory where the route-state package bundle will be written.')
    return parser


def run_route_state_package(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = Path(args.manifest).resolve()
    manifest = load_route_state_package_manifest(manifest_path)
    compilation = compile_route_state_package(
        manifest,
        manifest_base_dir=manifest_path.parent,
    )
    written_files = write_route_state_package_bundle(
        args.output_dir,
        compilation=compilation,
        metadata={
            'runner': 'backend/scripts/run_route_state_package.py',
            'manifest_path': str(manifest_path),
        },
    )
    return build_route_state_package_summary(
        compilation,
        output_dir=args.output_dir,
        bundle_manifest=written_files['bundle_manifest'],
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_route_state_package(args)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
