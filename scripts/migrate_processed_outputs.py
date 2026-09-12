"""Migrate legacy generated outputs into the canonical flat type layout.

Run without ``--apply`` for a dry run. Source recordings in ``data/raw`` are
outside this script's scope and are never touched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seima_mocap.output_layout import CATEGORIES, artifact_path, ensure_output_layout


PIPELINES = {
    "ball_tracking_real": "ball_tracking",
    "ball_tracking_probe": "ball_tracking_probe",
    "height_calibration_pilot": "height_calibration",
    "kinetics_short_clips": "planar_kinetics",
    "left_player_batch": "left_player",
    "left_player_pilot": "left_player_pilot",
}

ARTIFACT_ALIASES = {
    "batch_analysis": "analysis",
    "batch_motion_summary": "motion_summary",
    "batch_summary": "summary",
    "ball_tracking_diagnostic": "diagnostic",
    "kinetics_diagnostics": "kinetics",
    "left_player_motion_annotated": "annotated",
    "left_player_motion_silent": "silent",
    "pose_metrics": "frame_metrics",
    "scenario_summaries": "scenario_summaries",
    "semantic_events_all_videos": "semantic_events",
    "two_players_metrics": "frame_metrics",
    "stroke_dataset": "strokes",
}


def category_for(path: Path) -> str:
    name = path.stem.lower()
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        return "videos"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "diagnostics"
    if suffix in {".npz", ".npy"}:
        return "arrays"
    if suffix == ".md":
        return "reports"
    if suffix in {".log", ".txt"}:
        return "logs"
    if suffix == ".csv":
        if "event" in name:
            return "events"
        if "dataset" in name:
            return "datasets"
        if "summary" in name or "summaries" in name:
            return "summaries"
        return "metrics"
    if suffix == ".json":
        return "summaries" if "summary" in name or "summaries" in name else "metadata"
    return "metadata"


def root_file_context(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    for marker, pipeline, artifact in (
        ("_two_players_annotated", "two_players", "annotated"),
        ("_two_players_metrics", "two_players", "frame_metrics"),
        ("_two_players_summary", "two_players", "summary"),
        ("_pose_annotated", "pose", "annotated"),
        ("_pose_metrics", "pose", "frame_metrics"),
        ("_pose_summary", "pose", "summary"),
    ):
        if stem.endswith(marker):
            return stem[: -len(marker)], pipeline, artifact
    return "project", "legacy", stem


def legacy_context(relative: Path) -> tuple[str, str, str]:
    if len(relative.parts) == 1:
        return root_file_context(relative)
    family = relative.parts[0]
    pipeline = PIPELINES.get(family, family)
    if len(relative.parts) >= 3:
        source = relative.parts[1]
    elif family == "ball_tracking_probe":
        source = "probe"
    else:
        source = "batch"
    artifact = ARTIFACT_ALIASES.get(relative.stem, relative.stem.lower())
    if artifact == "report":
        artifact = "report"
    return source, pipeline, artifact


def plan(root: Path) -> list[tuple[Path, Path]]:
    root = root.resolve()
    moves: list[tuple[Path, Path]] = []
    destinations: set[Path] = set()
    for source in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = source.relative_to(root)
        if relative.parts[0] in CATEGORIES or relative.name in {".gitkeep", "README.md"}:
            continue
        source_name, pipeline, artifact = legacy_context(relative)
        target = artifact_path(root, category_for(source), source_name, pipeline, artifact, source.suffix)
        if target in destinations:
            raise FileExistsError(f"Two legacy files map to {target}")
        if target.exists():
            raise FileExistsError(f"Destination already exists: {target}")
        destinations.add(target)
        moves.append((source, target))
    return moves


def apply_moves(root: Path, moves: list[tuple[Path, Path]]) -> None:
    resolved_root = root.resolve()
    for source, target in moves:
        if resolved_root not in source.resolve().parents or resolved_root not in target.resolve().parents:
            raise ValueError(f"Refusing path outside processed root: {source} -> {target}")
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()),
                            key=lambda path: len(path.parts), reverse=True):
        if directory.name not in CATEGORIES:
            try:
                directory.rmdir()
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--apply", action="store_true", help="Perform the planned moves")
    args = parser.parse_args()
    ensure_output_layout(args.root)
    moves = plan(args.root)
    for source, target in moves:
        print(f"{source.relative_to(args.root)} -> {target.relative_to(args.root)}")
    print(f"Planned files: {len(moves)}")
    if args.apply:
        apply_moves(args.root, moves)
        print("Migration completed")
    elif moves:
        print("Dry run only; pass --apply to move files")


if __name__ == "__main__":
    main()
