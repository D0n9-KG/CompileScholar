from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.research_logic import (  # noqa: E402
    build_bounded_packet_audit_summary,
    load_bounded_packet_assembly_manifest,
    load_route_packet,
    render_bounded_packet_audit_report,
    validate_bounded_packet_assembly,
    write_bounded_packet_audit_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Audit one bounded RoutePacket plus companion support / alternative / held_out assembly manifest.'
    )
    parser.add_argument('--packet', required=True, help='Path to the RoutePacket JSON file to audit.')
    parser.add_argument(
        '--assembly-manifest',
        required=True,
        help='Path to the BoundedPacketAssemblyManifest JSON file layered on top of the packet.',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory where audit_summary.json, audit_inspection.json, and copied inputs will be written.',
    )
    parser.add_argument(
        '--report-md',
        help='Optional Markdown report path derived from the runtime audit summary and inspection payloads.',
    )
    return parser


def run_bounded_packet_audit(args: argparse.Namespace) -> dict[str, object]:
    packet_path = Path(args.packet).resolve()
    manifest_path = Path(args.assembly_manifest).resolve()
    route_packet = load_route_packet(packet_path)
    assembly_manifest = load_bounded_packet_assembly_manifest(manifest_path)
    audit = validate_bounded_packet_assembly(route_packet, assembly_manifest)

    report_markdown = (
        render_bounded_packet_audit_report(route_packet, assembly_manifest, audit)
        if str(args.report_md or '').strip()
        else None
    )
    written_files = write_bounded_packet_audit_bundle(
        args.output_dir,
        route_packet=route_packet,
        assembly_manifest=assembly_manifest,
        audit=audit,
        metadata={
            'runner': 'backend/scripts/run_bounded_packet_audit.py',
            'packet_path': str(packet_path),
            'assembly_manifest_path': str(manifest_path),
        },
        report_markdown=report_markdown,
    )

    summary = build_bounded_packet_audit_summary(
        route_packet=route_packet,
        assembly_manifest=assembly_manifest,
        audit=audit,
    )
    summary['output_dir'] = str(Path(args.output_dir).resolve())
    summary['bundle_manifest'] = str(written_files['bundle_manifest'].resolve())
    summary['audit_summary_file'] = str(written_files['audit_summary'].resolve())
    summary['audit_inspection_file'] = str(written_files['audit_inspection'].resolve())

    if report_markdown is not None:
        report_path = Path(args.report_md).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_markdown, encoding='utf-8')
        summary['report_md'] = str(report_path)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_bounded_packet_audit(args)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({'error': str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
