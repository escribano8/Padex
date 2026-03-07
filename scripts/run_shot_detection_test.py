"""
Project: Padex
File Created: 2026-03-06
Author: Xingnan Zhu
File Name: run_shot_detection_test.py
Description:
    End-to-end test script: runs tracking + bounce detection + shot
    classification on a video, then exports an annotated video with
    shot type labels overlaid on each frame.

    Output goes to output/<YYYYMMDD_HHMMSS>/ under the project root.

Usage:
    uv run python scripts/run_shot_detection_test.py [VIDEO_PATH]

    Default video: assets/processed/video/ThreeTest.mp4
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shot_detection_test")

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent


def main(video_path: Path, calibration_path: Path | None = None) -> None:
    import json

    from padex.events.bounce import BounceDetector
    from padex.events.shot import PoseBasedShotTypeClassifier, ShotDetector
    from padex.io.video import VideoReader, VideoWriter
    from padex.schemas.tracking import CourtCalibration
    from padex.tracking.pipeline import TrackingPipeline
    from padex.viz.frame import FrameAnnotator

    # -----------------------------------------------------------------------
    # Output directory: output/<timestamp>/
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "output" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # -----------------------------------------------------------------------
    # Load manual calibration if provided
    # -----------------------------------------------------------------------
    manual_cal = None
    if calibration_path is not None and calibration_path.exists():
        with open(calibration_path) as f:
            cal_data = json.load(f)
        manual_cal = CourtCalibration(**cal_data)
        logger.info(
            "Loaded calibration: %s (error=%.3fm)",
            calibration_path.name,
            manual_cal.reprojection_error or -1,
        )
    else:
        logger.info("No calibration file provided — will attempt auto-detection")

    # -----------------------------------------------------------------------
    # Stage 1: Tracking pipeline (with pickle cache to skip on re-runs)
    # -----------------------------------------------------------------------
    import pickle

    cache_path = PROJECT_ROOT / "output" / f"{video_path.stem}_tracking_cache.pkl"

    if cache_path.exists():
        logger.info("=== Stage 1: Loading cached tracking results ===")
        with open(cache_path, "rb") as f:
            tracking = pickle.load(f)
        logger.info(
            "Loaded from cache: %d player frames, %d ball frames",
            len(tracking.player_frames),
            len(tracking.ball_frames),
        )
    else:
        logger.info("=== Stage 1: Running tracking pipeline ===")
        pipeline = TrackingPipeline(
            video_path=video_path,
            enable_pose=True,
            manual_calibration=manual_cal,
        )
        tracking = pipeline.run()
        logger.info(
            "Tracking done: %d player frames, %d ball frames",
            len(tracking.player_frames),
            len(tracking.ball_frames),
        )
        with open(cache_path, "wb") as f:
            pickle.dump(tracking, f)
        logger.info("Tracking results cached: %s", cache_path)

    # -----------------------------------------------------------------------
    # Stage 2: Bounce detection
    # -----------------------------------------------------------------------
    logger.info("=== Stage 2: Bounce detection ===")
    bounce_detector = BounceDetector()
    bounces = bounce_detector.detect_bounces(tracking.ball_frames, tracking.calibration)
    logger.info("Detected %d bounces", len(bounces))
    for b in bounces[:10]:
        logger.info("  Bounce: %s @ %.0f ms", b.type.value, b.timestamp_ms or 0)

    # -----------------------------------------------------------------------
    # Stage 3: Shot detection + classification
    # -----------------------------------------------------------------------
    logger.info("=== Stage 3: Shot detection ===")
    shot_detector = ShotDetector(
        shot_type_classifier=PoseBasedShotTypeClassifier(),
    )
    shots = shot_detector.detect_shots(
        player_frames=tracking.player_frames,
        ball_frames=tracking.ball_frames,
        bounces=bounces,
    )
    logger.info("Detected %d shots", len(shots))
    for s in shots:
        logger.info(
            "  Shot %-20s | player=%-8s | type=%-18s | conf=%.2f | ts=%.0f ms",
            s.shot_id, s.player_id, s.shot_type.value, s.confidence, s.timestamp_ms,
        )

    # -----------------------------------------------------------------------
    # Stage 4: Build frame → shot lookup
    # Each shot is displayed for SHOT_DISPLAY_MS after contact
    # -----------------------------------------------------------------------
    SHOT_DISPLAY_MS = 1500.0

    # Map timestamp_ms → shot
    shot_events = sorted(shots, key=lambda s: s.timestamp_ms)

    def get_active_shot(timestamp_ms: float):
        """Return the most recent shot that is still within display window."""
        active = None
        for s in shot_events:
            if s.timestamp_ms <= timestamp_ms <= s.timestamp_ms + SHOT_DISPLAY_MS:
                active = s
        return active

    # -----------------------------------------------------------------------
    # Stage 5: Build frame → player/ball lookups
    # -----------------------------------------------------------------------
    player_lookup: dict[int, list] = defaultdict(list)
    for pf in tracking.player_frames:
        player_lookup[pf.frame_id].append(pf)

    ball_lookup: dict[int, object] = {}
    for bf in tracking.ball_frames:
        ball_lookup[bf.frame_id] = bf

    # -----------------------------------------------------------------------
    # Stage 6: Export annotated video
    # -----------------------------------------------------------------------
    logger.info("=== Stage 6: Exporting annotated video ===")
    out_video = out_dir / "shot_detection.mp4"
    annotator = FrameAnnotator()

    with VideoReader(video_path) as reader:
        fps = reader.fps
        w, h = reader.frame_size
        logger.info("Video: %dx%d @ %.1f fps, %d frames", w, h, fps, reader.frame_count)

        with VideoWriter(out_video, fps=fps, frame_size=(w, h)) as writer:
            shot_counts: dict[str, int] = defaultdict(int)
            for s in shots:
                shot_counts[s.shot_type.value] += 1

            for frame_id, timestamp_ms, frame in reader.frames():
                player_frames_here = player_lookup.get(frame_id, [])
                ball_frame_here = ball_lookup.get(frame_id)
                active_shot = get_active_shot(timestamp_ms)

                stats = {
                    "Frame": frame_id,
                    "Shots": len(shots),
                }
                # Show top shot types in stats
                for shot_type, count in sorted(
                    shot_counts.items(), key=lambda x: -x[1]
                )[:4]:
                    stats[shot_type[:12]] = count

                annotator.annotate_frame(
                    frame=frame,
                    frame_id=frame_id,
                    player_frames=player_frames_here,
                    ball_frame=ball_frame_here,
                    calibration=tracking.calibration,
                    shot=active_shot,
                    stats=stats,
                )
                writer.write(frame)

                if frame_id % 100 == 0:
                    logger.info("  Written frame %d", frame_id)

    logger.info("Annotated video saved: %s", out_video)

    # -----------------------------------------------------------------------
    # Stage 7: Print summary report
    # -----------------------------------------------------------------------
    logger.info("=== Summary ===")
    logger.info("Total shots detected: %d", len(shots))
    logger.info("Shot type breakdown:")
    for shot_type, count in sorted(shot_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-20s: %d", shot_type, count)
    logger.info("Bounces: %d", len(bounces))
    logger.info("Output: %s", out_dir)


def run_interactive_calibration(video_path: Path) -> Path | None:
    """Launch interactive calibration UI, save result next to video, return JSON path."""
    import json as _json

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from manual_calibrate import FrameSelector, KeypointLabeler  # type: ignore[import]

    from padex.tracking.court import CourtDetector

    logger.info("=== Interactive Court Calibration ===")
    logger.info("Phase 1: Browse to a frame with clear court lines, then press Enter")

    selector = FrameSelector(video_path)
    result = selector.run()
    selector.release()
    if result is None:
        logger.warning("Calibration cancelled.")
        return None
    frame, frame_id = result
    h, w = frame.shape[:2]
    logger.info("Selected frame %d (%dx%d)", frame_id, w, h)

    logger.info("Phase 2: Click the 12 court keypoints  (N=skip, Z=undo, Enter=finish)")
    labeler = KeypointLabeler(frame)
    keypoints = labeler.run()
    if keypoints is None:
        logger.warning("Calibration cancelled.")
        return None

    logger.info("Labeled %d keypoints — computing homography…", len(keypoints))
    try:
        calibration = CourtDetector.manual_calibration(
            keypoints_px=keypoints,
            frame_width=w,
            frame_height=h,
        )
    except ValueError as exc:
        logger.error("Calibration failed: %s", exc)
        return None

    logger.info("Reprojection error: %.4f m", calibration.reprojection_error or -1)

    cal_path = video_path.with_name(video_path.stem + "_calibration.json")
    cal_dict = calibration.model_dump()
    cal_dict["source_video"] = str(video_path)
    cal_dict["source_frame_id"] = frame_id
    cal_dict["labeled_keypoints"] = {k: list(v) for k, v in keypoints.items()}
    with open(cal_path, "w") as f:
        _json.dump(cal_dict, f, indent=2)
    logger.info("Calibration saved: %s", cal_path)
    return cal_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    else:
        video = PROJECT_ROOT / "assets" / "processed" / "video" / "ThreeTest.mp4"

    if not video.exists():
        logger.error("Video not found: %s", video)
        sys.exit(1)

    # Look for calibration: CLI arg → sibling JSON → interactive calibration
    if len(sys.argv) > 2:
        cal = Path(sys.argv[2])
    else:
        cal_sibling = video.with_name(video.stem + "_calibration.json")
        if cal_sibling.exists():
            cal = cal_sibling
            logger.info("Using existing calibration: %s", cal)
        else:
            logger.info("No calibration file found — launching interactive calibration")
            cal = run_interactive_calibration(video)
            if cal is None:
                logger.error("Calibration required. Exiting.")
                sys.exit(1)

    main(video, calibration_path=cal)
