"""
Project: Padex
File Created: 2026-03-07
Author: Xingnan Zhu
File Name: calibration.py
Description:
    Interactive court calibration tools.

    Provides FrameSelector and KeypointLabeler for manual court keypoint
    labeling, plus a convenience function interactive_calibrate() that
    runs the full calibration workflow.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from padex.schemas.tracking import CourtCalibration
from padex.tracking.court import COURT_MODEL, CourtDetector

logger = logging.getLogger(__name__)

# Keypoints in labeling order: corners first, then net, then service lines.
KEYPOINT_ORDER = [
    "bottom_left",
    "bottom_right",
    "top_left",
    "top_right",
    "net_left",
    "net_right",
    "service_near_left",
    "service_near_center",
    "service_near_right",
    "service_far_left",
    "service_far_center",
    "service_far_right",
]

# Color per keypoint group (BGR)
KEYPOINT_COLORS = {
    "bottom_left": (0, 0, 255),
    "bottom_right": (0, 68, 255),
    "top_left": (255, 0, 0),
    "top_right": (255, 68, 68),
    "net_left": (0, 255, 255),
    "net_right": (0, 200, 200),
    "service_near_left": (0, 255, 0),
    "service_near_center": (0, 200, 0),
    "service_near_right": (0, 150, 0),
    "service_far_left": (255, 0, 255),
    "service_far_center": (200, 0, 200),
    "service_far_right": (150, 0, 150),
}

# Court meter coords for display
KEYPOINT_METERS = COURT_MODEL.KEYPOINTS


class FrameSelector:
    """Phase 1: Let the user browse frames and pick a good one."""

    def __init__(self, video_path: str | Path) -> None:
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_id = 0
        self.frame: np.ndarray | None = None

    def run(self) -> tuple[np.ndarray, int] | None:
        """Show frames, return (frame, frame_id) or None if user quits."""
        self._read_frame()

        cv2.namedWindow("Select Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Frame", 1280, 720)

        while True:
            display = self._draw_overlay()
            cv2.imshow("Select Frame", display)

            key = cv2.waitKey(0) & 0xFF

            if key == ord("q") or key == 27:  # Esc
                cv2.destroyAllWindows()
                return None
            elif key == 13:  # Enter
                cv2.destroyAllWindows()
                return self.frame.copy(), self.frame_id
            elif key == ord("d") or key == 83:  # right arrow
                self._seek(90)
            elif key == ord("a") or key == 81:  # left arrow
                self._seek(-90)
            elif key == ord("w") or key == 82:  # up arrow
                self._seek(900)
            elif key == ord("s") or key == 84:  # down arrow
                self._seek(-900)

    def _seek(self, delta: int) -> None:
        self.frame_id = max(0, min(self.total_frames - 1, self.frame_id + delta))
        self._read_frame()

    def _read_frame(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame

    def _draw_overlay(self) -> np.ndarray:
        display = self.frame.copy()
        h, w = display.shape[:2]
        timestamp = self.frame_id / self.fps if self.fps > 0 else 0

        info = f"Frame {self.frame_id}/{self.total_frames}  |  {timestamp:.1f}s  |  {w}x{h}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        controls = "A/D: prev/next  W/S: jump  Enter: confirm  Q: quit"
        cv2.putText(display, controls, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return display

    def release(self) -> None:
        self.cap.release()


class KeypointLabeler:
    """Phase 2: Let the user click keypoints on the selected frame."""

    def __init__(self, frame: np.ndarray) -> None:
        self.original = frame.copy()
        self.frame_h, self.frame_w = frame.shape[:2]
        self.current_idx = 0
        self.labeled: dict[str, tuple[float, float]] = {}
        self.click_pos: tuple[int, int] | None = None

    def run(self) -> dict[str, tuple[float, float]] | None:
        """Run the labeling loop. Returns keypoint dict or None if quit."""
        cv2.namedWindow("Label Keypoints", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Label Keypoints", 1280, 720)
        cv2.setMouseCallback("Label Keypoints", self._on_mouse)

        while True:
            display = self._draw_overlay()
            cv2.imshow("Label Keypoints", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == 27:
                cv2.destroyAllWindows()
                return None

            elif key == ord("z"):
                self._undo()

            elif key == ord("n"):
                if self.current_idx < len(KEYPOINT_ORDER):
                    name = KEYPOINT_ORDER[self.current_idx]
                    logger.info("Skipped: %s", name)
                    self.current_idx += 1

            elif key == 13:
                if len(self.labeled) >= 4:
                    cv2.destroyAllWindows()
                    return self.labeled
                else:
                    logger.warning("Need at least 4 keypoints, have %d", len(self.labeled))

            if self.click_pos is not None and self.current_idx < len(KEYPOINT_ORDER):
                name = KEYPOINT_ORDER[self.current_idx]
                self.labeled[name] = (float(self.click_pos[0]), float(self.click_pos[1]))
                logger.info("Placed: %s at (%d, %d)", name, self.click_pos[0], self.click_pos[1])
                self.current_idx += 1
                self.click_pos = None

            if self.current_idx >= len(KEYPOINT_ORDER):
                cv2.destroyAllWindows()
                return self.labeled

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_pos = (x, y)

    def _undo(self) -> None:
        if self.current_idx > 0:
            self.current_idx -= 1
            while self.current_idx >= 0:
                name = KEYPOINT_ORDER[self.current_idx]
                if name in self.labeled:
                    del self.labeled[name]
                    logger.info("Undone: %s", name)
                    break
                self.current_idx -= 1
                if self.current_idx < 0:
                    self.current_idx = 0
                    break

    def _draw_overlay(self) -> np.ndarray:
        display = self.original.copy()

        for name, (px, py) in self.labeled.items():
            color = KEYPOINT_COLORS[name]
            ix, iy = int(px), int(py)
            cv2.circle(display, (ix, iy), 6, color, -1)
            cv2.circle(display, (ix, iy), 8, (255, 255, 255), 1)

            meters = KEYPOINT_METERS[name]
            label = f"{name} ({meters[0]},{meters[1]})"
            cv2.putText(display, label, (ix + 10, iy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if self.current_idx < len(KEYPOINT_ORDER):
            name = KEYPOINT_ORDER[self.current_idx]
            meters = KEYPOINT_METERS[name]
            prompt = f"Click: {name}  ({meters[0]}m, {meters[1]}m)  [{self.current_idx + 1}/12]"
            color = KEYPOINT_COLORS[name]
        else:
            prompt = "All keypoints placed! Press Enter to confirm."
            color = (0, 255, 0)

        cv2.rectangle(display, (0, 0), (self.frame_w, 40), (0, 0, 0), -1)
        cv2.putText(display, prompt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        status = f"Labeled: {len(self.labeled)}/12  |  Z: undo  N: skip  Enter: finish  Q: quit"
        cv2.rectangle(display, (0, self.frame_h - 35), (self.frame_w, self.frame_h), (0, 0, 0), -1)
        cv2.putText(display, status, (10, self.frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return display


def verify_calibration(
    frame: np.ndarray,
    calibration_data: dict,
) -> None:
    """Show the frame with all 12 keypoints projected back for visual verification."""
    H = np.array(calibration_data["homography_matrix"])
    H_inv = np.linalg.inv(H)

    display = frame.copy()

    for name, (mx, my) in KEYPOINT_METERS.items():
        pt = np.array([[[mx, my]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(pt, H_inv)
        px, py = int(projected[0, 0, 0]), int(projected[0, 0, 1])
        color = KEYPOINT_COLORS.get(name, (255, 255, 255))

        cv2.circle(display, (px, py), 5, color, -1)
        cv2.circle(display, (px, py), 7, (255, 255, 255), 1)
        cv2.putText(display, name, (px + 8, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    for kp_a, kp_b in COURT_MODEL.LINES:
        ma = KEYPOINT_METERS[kp_a]
        mb = KEYPOINT_METERS[kp_b]
        pa = cv2.perspectiveTransform(np.array([[[ma[0], ma[1]]]], dtype=np.float64), H_inv)
        pb = cv2.perspectiveTransform(np.array([[[mb[0], mb[1]]]], dtype=np.float64), H_inv)
        pt_a = (int(pa[0, 0, 0]), int(pa[0, 0, 1]))
        pt_b = (int(pb[0, 0, 0]), int(pb[0, 0, 1]))
        cv2.line(display, pt_a, pt_b, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
    msg = f"Verification  |  Reprojection error: {calibration_data['reprojection_error']:.4f}m  |  Press any key to close"
    cv2.putText(display, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.namedWindow("Verify Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Verify Calibration", 1280, 720)
    cv2.imshow("Verify Calibration", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def interactive_calibrate(
    video_path: str | Path,
    save_path: str | Path | None = None,
) -> CourtCalibration | None:
    """Run the full interactive calibration workflow.

    Opens a video, lets the user select a frame and click court keypoints,
    then computes the homography and optionally saves to JSON.

    Args:
        video_path: Path to the video file.
        save_path: Where to save the calibration JSON. Defaults to
            ``<video_stem>_calibration.json`` next to the video.

    Returns:
        CourtCalibration if successful, None if cancelled.
    """
    video_path = Path(video_path)
    if save_path is None:
        save_path = video_path.with_name(video_path.stem + "_calibration.json")
    else:
        save_path = Path(save_path)

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

    logger.info("Phase 2: Click the 12 court keypoints (N=skip, Z=undo, Enter=finish)")
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

    cal_dict = calibration.model_dump()
    cal_dict["source_video"] = str(video_path)
    cal_dict["source_frame_id"] = frame_id
    cal_dict["labeled_keypoints"] = {k: list(v) for k, v in keypoints.items()}
    with open(save_path, "w") as f:
        json.dump(cal_dict, f, indent=2)
    logger.info("Calibration saved: %s", save_path)

    return calibration
