"""
Project: Padex
File Created: 2026-03-07
Author: Xingnan Zhu
File Name: cli.py
Description:
    Command-line interface for Padex.

Usage:
    padex process VIDEO [--calibration CAL] [--output DIR] [--no-cache]
    padex calibrate VIDEO [-o OUTPUT]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_process(args: argparse.Namespace) -> None:
    """Run the full Padex pipeline."""
    from padex.calibration import interactive_calibrate
    from padex.pipeline import Padex

    calibration = args.calibration

    # If no calibration provided, check for sibling file; if missing, run interactive
    if calibration is None:
        sibling = args.video.with_name(args.video.stem + "_calibration.json")
        if sibling.exists():
            calibration = sibling
        else:
            logger = logging.getLogger("padex.cli")
            logger.info("No calibration found — launching interactive calibration")
            cal = interactive_calibrate(args.video)
            if cal is None:
                logger.error("Calibration required. Exiting.")
                sys.exit(1)
            calibration = args.video.with_name(args.video.stem + "_calibration.json")

    padex = Padex(
        video_path=args.video,
        calibration=calibration,
        cache_tracking=not args.no_cache,
    )
    result = padex.run()

    # Summary
    logger = logging.getLogger("padex.cli")
    logger.info("Total shots detected: %d", len(result.shots))
    logger.info("Total bounces detected: %d", len(result.bounces))
    for s in result.shots:
        logger.info(
            "  %-18s | player=%-8s | conf=%.2f",
            s.shot_type.value, s.player_id, s.confidence,
        )

    # Export annotated video unless --no-export
    if not args.no_export:
        output_dir = Path(args.output) if args.output else Path("output")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = output_dir / timestamp
        out_video = out_dir / "shot_detection.mp4"

        padex.export_video(result, out_video)
        logger.info("Output: %s", out_dir)


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Run interactive court calibration."""
    from padex.calibration import interactive_calibrate

    cal = interactive_calibrate(
        video_path=args.video,
        save_path=args.output,
    )
    if cal is None:
        sys.exit(1)


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        prog="padex",
        description="Padex — padel video analytics toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # padex process
    p_process = subparsers.add_parser(
        "process", help="Run the full analysis pipeline on a video"
    )
    p_process.add_argument("video", type=Path, help="Path to video file")
    p_process.add_argument(
        "--calibration", "-c", type=Path, default=None,
        help="Path to court calibration JSON",
    )
    p_process.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory (default: output/)",
    )
    p_process.add_argument(
        "--no-cache", action="store_true",
        help="Disable tracking cache (re-run tracking from scratch)",
    )
    p_process.add_argument(
        "--no-export", action="store_true",
        help="Skip annotated video export",
    )
    p_process.set_defaults(func=cmd_process)

    # padex calibrate
    p_cal = subparsers.add_parser(
        "calibrate", help="Interactive court calibration"
    )
    p_cal.add_argument("video", type=Path, help="Path to video file")
    p_cal.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output JSON path (default: <video_stem>_calibration.json)",
    )
    p_cal.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
