"""
Project: Padex
File Created: 2026-03-05
Author: Xingnan Zhu
File Name: manual_calibrate.py
Description:
    Example script for interactive manual court calibration.

    This is a thin wrapper around padex.calibration.

Usage:
    python scripts/manual_calibrate.py assets/raw/video/match.mp4
    python scripts/manual_calibrate.py assets/raw/video/match.mp4 -o calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from padex.calibration import (
    FrameSelector,
    KeypointLabeler,
    interactive_calibrate,
    verify_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive manual court calibration tool."
    )
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output JSON path (default: <video_stem>_calibration.json)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        logger.error("Video not found: %s", args.video)
        sys.exit(1)

    output_path = args.output or Path(f"{args.video.stem}_calibration.json")

    calibration = interactive_calibrate(
        video_path=args.video,
        save_path=output_path,
    )

    if calibration is None:
        logger.info("Cancelled.")
        sys.exit(1)

    # Visual verification
    logger.info("Verify — court lines projected onto frame")
    with open(output_path) as f:
        cal_dict = json.load(f)

    import cv2
    selector = FrameSelector(args.video)
    # Seek to the calibration source frame
    source_frame_id = cal_dict.get("source_frame_id", 0)
    selector.frame_id = source_frame_id
    selector._read_frame()
    verify_calibration(selector.frame, cal_dict)
    selector.release()

    # Print usage hint
    print("\n--- How to use this calibration ---")
    print(f"""
import padex

result = padex.process("{args.video}", calibration="{output_path}")
print(f"Found {{len(result.shots)}} shots")
""")


if __name__ == "__main__":
    main()
