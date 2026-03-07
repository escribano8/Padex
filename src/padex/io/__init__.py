"""
Project: Padex
File Created: 2026-03-03
Author: Xingnan Zhu
File Name: __init__.py
Description:
    Data I/O — readers and writers for Parquet, JSONL, and video.
"""

from padex.io.video import VideoReader, VideoWriter

__all__ = ["VideoReader", "VideoWriter"]
