"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: test_player_detection.py
Description:
    Tests for player detection, tracking, and team classification.
    Unit tests use synthetic data; integration tests use video + YOLO model.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from padex.schemas.tracking import BoundingBox, PlayerFrame, Position2D
from padex.tracking.player import (
    JerseyColorTeamClassifier,
    PlayerDetector,
    RawDetection,
)

VIDEO_PATH = Path("assets/raw/video/TapiaChingottoLebronGalanHighlights_1080p.mp4")
MODEL_PATH = Path("yolo26m.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(
    x1=100, y1=50, x2=200, y2=300, conf=0.9, track_id=None, color=(0, 0, 255)
) -> RawDetection:
    """Create a RawDetection with a solid-colored crop."""
    crop = np.full((y2 - y1, x2 - x1, 3), color, dtype=np.uint8)
    return RawDetection(
        bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
        confidence=conf,
        track_id=track_id,
        crop=crop,
    )


# ---------------------------------------------------------------------------
# Unit tests — no video or model needed
# ---------------------------------------------------------------------------


class TestRawDetection:
    def test_creation(self):
        det = _make_detection()
        assert det.confidence == 0.9
        assert det.track_id is None

    def test_with_track_id(self):
        det = _make_detection(track_id=5)
        assert det.track_id == 5


class TestBboxFootPosition:
    def test_foot_position_center_bottom(self):
        bbox = BoundingBox(x1=100, y1=50, x2=200, y2=300)
        foot = PlayerDetector._bbox_foot_position(bbox)
        assert foot == (150.0, 300.0)

    def test_foot_position_asymmetric(self):
        bbox = BoundingBox(x1=0, y1=0, x2=80, y2=200)
        foot = PlayerDetector._bbox_foot_position(bbox)
        assert foot == (40.0, 200.0)


class TestGhostFiltering:
    def setup_method(self):
        # Use a mock strategy that won't be called
        self.detector = PlayerDetector.__new__(PlayerDetector)
        self.detector.MAX_PLAYERS = 4
        self.detector.COURT_MARGIN_M = 1.0

    def test_detection_inside_court_passes(self):
        # Identity homography: pixel coords == court coords
        H = np.eye(3)
        bbox = BoundingBox(x1=4.0, y1=0.0, x2=6.0, y2=10.0)
        result = self.detector._pixel_to_court(bbox, H)
        assert result is not None
        assert 0 <= result[0] <= 10
        assert 0 <= result[1] <= 20

    def test_detection_outside_court_rejected(self):
        H = np.eye(3)
        bbox = BoundingBox(x1=50.0, y1=0.0, x2=60.0, y2=100.0)
        result = self.detector._pixel_to_court(bbox, H)
        assert result is None

    def test_detection_within_margin_passes(self):
        H = np.eye(3)
        # Foot at (5.0, 10.5) — within 1m margin of court edge (y max = 20)
        bbox = BoundingBox(x1=4.0, y1=0.0, x2=6.0, y2=10.5)
        result = self.detector._pixel_to_court(bbox, H)
        assert result is not None

    def test_position_clamped_to_bounds(self):
        H = np.eye(3)
        # Foot at (5.0, 20.5) — within margin but > 20, should clamp
        bbox = BoundingBox(x1=4.0, y1=0.0, x2=6.0, y2=20.5)
        result = self.detector._pixel_to_court(bbox, H)
        assert result is not None
        assert result[1] == 20.0  # clamped


class TestJerseyColorClassifier:
    def test_two_distinct_colors(self):
        """Red and blue crops should be classified into different teams."""
        classifier = JerseyColorTeamClassifier(n_warmup_frames=1)

        # Create 2 red + 2 blue detections
        red = (0, 0, 255)
        blue = (255, 0, 0)
        dets = [
            _make_detection(track_id=1, color=red),
            _make_detection(track_id=2, color=blue),
            _make_detection(track_id=3, color=red),
            _make_detection(track_id=4, color=blue),
        ]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Warmup frame
        classifier.classify(dets, frame)
        # Actual classification
        result = classifier.classify(dets, frame)

        assert len(result) == 4
        # Red detections should share one team, blue another
        assert result[0] == result[2]  # both red
        assert result[1] == result[3]  # both blue
        assert result[0] != result[1]  # different teams

    def test_returns_empty_during_warmup(self):
        classifier = JerseyColorTeamClassifier(n_warmup_frames=5)
        dets = [_make_detection(track_id=1), _make_detection(track_id=2)]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = classifier.classify(dets, frame)
        assert result == {}

    def test_single_detection_returns_empty(self):
        classifier = JerseyColorTeamClassifier(n_warmup_frames=1)
        dets = [_make_detection(track_id=1)]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = classifier.classify(dets, frame)
        assert result == {}

    def test_small_crop_skipped(self):
        classifier = JerseyColorTeamClassifier(min_crop_height=100)
        det = _make_detection(y1=0, y2=10)  # only 10px tall
        dets = [det, _make_detection()]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = classifier.classify(dets, frame)
        assert result == {}

    def test_reset_clears_state(self):
        classifier = JerseyColorTeamClassifier(n_warmup_frames=1)
        dets = [_make_detection(track_id=1), _make_detection(track_id=2)]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        classifier.classify(dets, frame)
        classifier.classify(dets, frame)
        assert classifier._centers is not None

        classifier.reset()
        assert classifier._centers is None
        assert classifier._frame_count == 0

    def test_kmeans_2_basic(self):
        """Two well-separated clusters should be found."""
        cluster_a = np.random.randn(10, 4) + np.array([10, 0, 0, 0])
        cluster_b = np.random.randn(10, 4) + np.array([-10, 0, 0, 0])
        features = np.vstack([cluster_a, cluster_b]).astype(np.float32)

        centers = JerseyColorTeamClassifier._kmeans_2(features)
        assert centers.shape == (2, 4)
        # Centers should be near [10, 0, 0, 0] and [-10, 0, 0, 0]
        center_xs = sorted(centers[:, 0])
        assert center_xs[0] < -5
        assert center_xs[1] > 5


class TestPlayerIdFormatting:
    def test_tracked_id(self):
        assert f"P_{5:03d}" == "P_005"
        assert f"P_{12:03d}" == "P_012"

    def test_untracked_id(self):
        assert f"P_det_{0:03d}" == "P_det_000"
        assert f"P_det_{3:03d}" == "P_det_003"


# ---------------------------------------------------------------------------
# Integration tests — require video + YOLO model
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not VIDEO_PATH.exists() or not MODEL_PATH.exists(),
    reason="Test video or YOLO model not available",
)
class TestWithVideoAndModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        from padex.io.video import VideoReader

        self.reader = VideoReader(VIDEO_PATH)
        self.detector = PlayerDetector(model_path=str(MODEL_PATH))
        yield
        self.reader.__exit__(None, None, None)

    def test_detect_single_frame(self):
        frame = self.reader.read_frame(3000)
        results = self.detector.detect(frame, frame_id=3000, timestamp_ms=100000.0)
        assert isinstance(results, list)
        for pf in results:
            assert isinstance(pf, PlayerFrame)
            assert 0 < pf.confidence <= 1.0
            assert pf.keypoints == []

    def test_detect_returns_at_most_4(self):
        for fid, ts, frame in self.reader.frames(start_frame=3000, end_frame=3100):
            results = self.detector.detect(frame, frame_id=fid, timestamp_ms=ts)
            assert len(results) <= 4

    def test_tracking_stable_ids(self):
        ids_per_frame = []
        for fid, ts, frame in self.reader.frames(start_frame=3000, end_frame=3030):
            results = self.detector.detect_and_track(
                frame, frame_id=fid, timestamp_ms=ts
            )
            ids_per_frame.append({pf.player_id for pf in results})

        # Most IDs should persist across adjacent frames
        for i in range(1, len(ids_per_frame)):
            if ids_per_frame[i] and ids_per_frame[i - 1]:
                overlap = ids_per_frame[i] & ids_per_frame[i - 1]
                assert len(overlap) >= 2

    def test_team_classification_two_teams(self):
        detector = PlayerDetector(model_path=str(MODEL_PATH))
        team_ids = set()
        for fid, ts, frame in self.reader.frames(start_frame=3000, end_frame=3100):
            results = detector.detect_and_track(
                frame, frame_id=fid, timestamp_ms=ts
            )
            for pf in results:
                if pf.team_id is not None:
                    team_ids.add(pf.team_id)
        assert len(team_ids) == 2

    def test_court_position_with_homography(self):
        from padex.tracking.court import CourtDetector

        court_det = CourtDetector()
        frame = self.reader.read_frame(3000)
        cal = court_det.calibrate_frame(frame)
        if cal is None:
            pytest.skip("Could not calibrate this frame")
        H = np.array(cal.homography_matrix)
        results = self.detector.detect(
            frame, frame_id=3000, timestamp_ms=100000.0, homography_matrix=H
        )
        for pf in results:
            if pf.position is not None:
                assert 0 <= pf.position.x <= 10
                assert 0 <= pf.position.y <= 20
