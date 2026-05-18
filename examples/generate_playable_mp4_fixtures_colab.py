"""
Generate playable H.264 .mp4 test videos for Track 1.

Best place to run: Google Colab.

What it creates:
  examples/generated_mp4_fixtures/normal/licensed_master_clean.mp4
  examples/generated_mp4_fixtures/normal/original_upload_clean.mp4
  examples/generated_mp4_fixtures/defective/watermark_removal_attempt.mp4
  examples/generated_mp4_fixtures/defective/frame_level_duplication.mp4
  examples/generated_mp4_fixtures/defective/reencoding_licensed_clip_bypass.mp4

The final videos are H.264 + yuv420p MP4 files, which are broadly playable
in Windows Media Player, VLC, Chrome, and Colab preview.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("examples/generated_mp4_fixtures")
WORK = ROOT / "_frames"
NORMAL = ROOT / "normal"
DEFECTIVE = ROOT / "defective"

WIDTH = 1280
HEIGHT = 720
FPS = 24
DURATION_SECONDS = 8
FRAME_COUNT = FPS * DURATION_SECONDS
WATERMARK_BOX = (958, 54, 250, 82)
HASH_BITS = 64


def ensure_dirs() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    for path in (WORK, NORMAL, DEFECTIVE):
        path.mkdir(parents=True, exist_ok=True)


def draw_watermark(frame: np.ndarray) -> None:
    x, y, w, h = WATERMARK_BOX
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.rectangle(frame, (x + 16, y + 18), (x + 55, y + 58), (35, 75, 160), -1)
    cv2.rectangle(frame, (x + 70, y + 18), (x + 109, y + 58), (35, 75, 160), -1)
    cv2.rectangle(frame, (x + 124, y + 18), (x + 163, y + 58), (35, 75, 160), -1)
    cv2.putText(frame, "LICENSED", (x + 18, y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def draw_tamper_patch(frame: np.ndarray) -> None:
    x, y, w, h = WATERMARK_BOX
    patch = frame[y : y + h, x : x + w].copy()
    patch = cv2.GaussianBlur(patch, (41, 41), 0)
    frame[y : y + h, x : x + w] = patch
    cv2.rectangle(frame, (x + 8, y + 8), (x + w - 8, y + h - 8), (185, 185, 185), -1)
    cv2.rectangle(frame, (x + 26, y + 27), (x + w - 26, y + 45), (155, 155, 155), -1)
    cv2.putText(frame, "REMOVED", (x + 44, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 80, 80), 2)


def bits_to_hex(bits: np.ndarray) -> str:
    bit_string = "".join("1" if bit else "0" for bit in bits.astype(bool).flatten())
    return f"{int(bit_string, 2):0{HASH_BITS // 4}x}"


def grayscale(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def phash(frame: np.ndarray) -> str:
    gray = cv2.resize(grayscale(frame), (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(gray))
    low_freq = dct[:8, :8]
    median = np.median(low_freq[1:, 1:])
    return bits_to_hex(low_freq > median)


def dhash(frame: np.ndarray) -> str:
    gray = cv2.resize(grayscale(frame), (9, 8), interpolation=cv2.INTER_AREA)
    return bits_to_hex(gray[:, 1:] > gray[:, :-1])


def whash(frame: np.ndarray) -> str:
    gray = cv2.resize(grayscale(frame), (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    rows = (gray[:, 0::2] + gray[:, 1::2]) / 2.0
    approx = (rows[0::2, :] + rows[1::2, :]) / 2.0
    median = np.median(approx)
    return bits_to_hex(approx > median)


def hsv_histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.flatten()


def chi_square_distance(left: np.ndarray, right: np.ndarray) -> float:
    denom = left + right + 1e-10
    return float(0.5 * np.sum(((left - right) ** 2) / denom))


def base_frame(i: int, label: str, watermark: str = "clean", tint: tuple[int, int, int] | None = None) -> np.ndarray:
    scene = i // (FPS * 2)
    t = i / FPS
    backgrounds = [
        (40, 90, 135),
        (46, 145, 105),
        (170, 88, 54),
        (80, 70, 145),
    ]
    frame = np.full((HEIGHT, WIDTH, 3), backgrounds[scene % len(backgrounds)], dtype=np.uint8)

    # Large moving foreground objects make the visual copy obvious.
    x = int(72 + (WIDTH - 300) * ((i % (FPS * 2)) / (FPS * 2)))
    y = int(205 + 50 * math.sin(i / 9))
    cv2.rectangle(frame, (x, y), (x + 180, y + 105), (245, 245, 245), -1)
    cv2.rectangle(frame, (x, y), (x + 180, y + 16), (25, 25, 25), -1)
    cv2.circle(frame, (WIDTH - 210 - (x // 5), 390), 62, (30, 35, 45), -1)
    cv2.circle(frame, (WIDTH - 210 - (x // 5), 390), 34, (245, 203, 70), -1)

    # Scene marker supports temporal fingerprinting.
    cv2.rectangle(frame, (58, 54), (335, 128), (255, 255, 255), -1)
    cv2.putText(frame, f"SCENE {scene + 1}", (78, 103), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (35, 35, 35), 3)
    cv2.putText(frame, label, (54, 662), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)

    if watermark == "clean":
        draw_watermark(frame)
    elif watermark == "tampered":
        draw_tamper_patch(frame)

    if tint is not None:
        overlay = np.full_like(frame, tint)
        frame = cv2.addWeighted(frame, 0.86, overlay, 0.14, 0)

    return frame


def make_frames(kind: str) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for i in range(FRAME_COUNT):
        if kind == "master":
            frame = base_frame(i, "NORMAL: LICENSED MASTER", watermark="clean")
        elif kind == "clean_upload":
            frame = base_frame(i, "NORMAL: CLEAN UPLOAD", watermark="clean")
        elif kind == "watermark_tamper":
            frame = base_frame(i, "DEFECT: WATERMARK REMOVAL ATTEMPT", watermark="tampered")
        elif kind == "frame_duplication":
            # Same visual content as master, but deliberately softened and re-scaled
            # to mimic copied frames after recompression/resolution changes.
            frame = base_frame(i, "DEFECT: FRAME-LEVEL DUPLICATION", watermark="clean")
            small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            frame = cv2.resize(small, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
        elif kind == "reencoding":
            # Same licensed sequence with timestamp drift, mild color shift, and
            # letterboxing: a classic re-encoding bypass fixture.
            shifted_i = min(FRAME_COUNT - 1, i + FPS // 2)
            frame = base_frame(
                shifted_i,
                "DEFECT: RE-ENCODED LICENSED CLIP",
                watermark="clean",
                tint=(18, 28, 44),
            )
            frame[:36, :] = 0
            frame[-36:, :] = 0
        else:
            raise ValueError(f"Unknown fixture kind: {kind}")
        frames.append(frame)
    return frames


def write_png_sequence(frames: list[np.ndarray], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        cv2.imwrite(str(folder / f"frame_{index:04d}.png"), frame)


def write_reference_assets(master_frames: list[np.ndarray]) -> None:
    x, y, w, h = WATERMARK_BOX
    watermark_patch = master_frames[0][y : y + h, x : x + w]
    cv2.imwrite(str(ROOT / "watermark_reference.png"), watermark_patch)

    sampled_indexes = list(range(0, len(master_frames), FPS))
    reference_hashes = {
        "frames": [
            {
                "asset_id": "licensed_master_clean",
                "timestamp_sec": index // FPS,
                "phash": phash(master_frames[index]),
                "dhash": dhash(master_frames[index]),
                "whash": whash(master_frames[index]),
            }
            for index in sampled_indexes
        ]
    }
    (ROOT / "reference_hashes_master.json").write_text(json.dumps(reference_hashes, indent=2), encoding="utf-8")

    sampled_frames = [master_frames[index] for index in sampled_indexes]
    hists = [hsv_histogram(frame) for frame in sampled_frames]
    cuts = [index for index in range(1, len(hists)) if chi_square_distance(hists[index - 1], hists[index]) >= 0.45]
    boundaries = [0, *cuts, len(sampled_frames)]
    scenes = []
    for start, end in zip(boundaries, boundaries[1:]):
        scene_frames = sampled_frames[start:end]
        midpoint_index = start + (len(scene_frames) // 2)
        scenes.append(
            {
                "duration_sec": float(end - start),
                "phash": phash(sampled_frames[midpoint_index]),
            }
        )

    licensed = {
        "clips": [
            {
                "clip_id": "licensed_master_clean",
                "scenes": scenes,
            }
        ]
    }
    (ROOT / "licensed_fingerprints_master.json").write_text(json.dumps(licensed, indent=2), encoding="utf-8")


def encode_h264_mp4(frame_folder: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        str(frame_folder / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-profile:v",
        "main",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def generate_fixture(name: str, kind: str, output_dir: Path) -> None:
    frame_folder = WORK / name
    frames = make_frames(kind)
    write_png_sequence(frames, frame_folder)
    encode_h264_mp4(frame_folder, output_dir / f"{name}.mp4")


def main() -> None:
    ensure_dirs()

    master_frames = make_frames("master")
    write_reference_assets(master_frames)

    generate_fixture("licensed_master_clean", "master", NORMAL)
    generate_fixture("original_upload_clean", "clean_upload", NORMAL)
    generate_fixture("watermark_removal_attempt", "watermark_tamper", DEFECTIVE)
    generate_fixture("frame_level_duplication", "frame_duplication", DEFECTIVE)
    generate_fixture("reencoding_licensed_clip_bypass", "reencoding", DEFECTIVE)

    manifest = {
        "format": "H.264 MP4, yuv420p",
        "resolution": f"{WIDTH}x{HEIGHT}",
        "fps": FPS,
        "duration_seconds": DURATION_SECONDS,
        "watermark_box_pixels": WATERMARK_BOX,
        "reference_assets": [
            "watermark_reference.png",
            "reference_hashes_master.json",
            "licensed_fingerprints_master.json",
        ],
        "normal_videos": [
            "normal/licensed_master_clean.mp4",
            "normal/original_upload_clean.mp4",
        ],
        "defective_videos": [
            "defective/watermark_removal_attempt.mp4",
            "defective/frame_level_duplication.mp4",
            "defective/reencoding_licensed_clip_bypass.mp4",
        ],
    }
    (ROOT / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.rmtree(WORK)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
