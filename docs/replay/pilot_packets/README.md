# Replay Pilot Packets

This directory stores committed packet-level artifacts for historical replay pilots.

What is committed here:

- canonical `RoutePacket` manifests
- human-readable packet selection notes
- packet-level README / audit context

What stays local-only:

- actual `PaperLogicTrace` JSON exports
- local corpus directories and network-share paths
- runtime replay bundles written by `backend/scripts/run_replay_pilot.py`

Design rule:

- repo files capture the bounded packet definition
- local CLI arguments provide the machine-specific trace source

The goal is to keep each replay pilot auditable and reproducible without hardcoding corpus paths into the repository.
