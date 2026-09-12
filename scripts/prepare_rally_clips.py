"""Propose, review and export fixed-camera rally clips from an original video."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seima_mocap.rally_clips import prepare_manifest, export_manifest
from seima_mocap.guided_review import review_manifest
from seima_mocap.rally_review import review_manifest_advanced
from seima_mocap.video_io import resolve_video_tool
from annotate_table_geometry import annotate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, nargs="?")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--advanced-review", action="store_true", help="Editor completo opcional")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--inactivity-s", type=float, default=1.5)
    parser.add_argument("--padding-s", type=float, default=.5)
    parser.add_argument("--activity-window-s", type=float, default=.8)
    parser.add_argument("--activity-fraction", type=float, default=.3)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    args = parser.parse_args()
    ffprobe = resolve_video_tool("ffprobe", args.ffprobe)
    ffmpeg = resolve_video_tool("ffmpeg", args.ffmpeg) if args.export or args.review or args.ffmpeg else None
    if not args.manifest.exists():
        if args.video is None:
            parser.error("video is required to prepare a new manifest")
        value = prepare_manifest(ROOT, args.video.resolve(), args.manifest, inactivity_s=args.inactivity_s,
                                 padding_s=args.padding_s, ffprobe=ffprobe, activity_window_s=args.activity_window_s,
                                 activity_fraction=args.activity_fraction)
        print(f"Proposed {len(value['rallies'])} rallies and {len(value['shots'])} shots. Review required.", flush=True)
    if args.advanced_review:
        review_manifest_advanced(args.manifest, ROOT, annotate)
    elif args.review:
        if not review_manifest(args.manifest, ROOT, annotate):
            print("Progreso guardado. Abra de nuevo el revisor para continuar.", flush=True)
            return
    if args.export:
        export_manifest(args.manifest, args.output_root, ffmpeg=ffmpeg, ffprobe=ffprobe)


if __name__ == "__main__":
    main()
