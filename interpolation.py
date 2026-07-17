"""Bayer (RGGB) demosaicing of a packed 10-bit RAW image — Python/numpy version.

Python has no direct OpenMP equivalent; the parallel speedup is achieved
instead with vectorised numpy operations, which run compiled C loops
(and use SIMD) rather than a Python loop per pixel.

The arithmetic (integer averages with truncation, mirrored borders) matches
interpolation.cpp exactly, so `said_py.raw` is byte-identical to `said.raw`.

Usage: python3 interpolation.py [raw_file]  (default: raw.RAW, 3264x2448)
"""

import sys
import time

import numpy as np

WIDTH = 3264
HEIGHT = 2448


def unpack_10bit(raw: np.ndarray, pixel_count: int) -> np.ndarray:
    """Every 5 bytes hold 4 pixels: 4 high bytes + 1 byte of 2-bit remainders."""
    groups = raw.reshape(-1, 5).astype(np.uint16)
    pixels = np.empty((groups.shape[0], 4), dtype=np.uint16)
    low = groups[:, 4]
    for k in range(4):
        pixels[:, k] = (groups[:, k] << 2) | ((low >> (2 * k)) & 3)
    return pixels.reshape(pixel_count)


def split_channels(bayer: np.ndarray):
    """Split the RGGB mosaic into R, G, B planes (missing pixels are 0)."""
    red = np.zeros_like(bayer)
    green = np.zeros_like(bayer)
    blue = np.zeros_like(bayer)
    red[0::2, 0::2] = bayer[0::2, 0::2]
    green[0::2, 1::2] = bayer[0::2, 1::2]
    green[1::2, 0::2] = bayer[1::2, 0::2]
    blue[1::2, 1::2] = bayer[1::2, 1::2]
    return red, green, blue


def interpolate_red(a: np.ndarray) -> None:
    """Red is known at (even row, even col); values are 10-bit so uint16 sums never overflow."""
    # horizontal pass: odd columns of even rows, last column by replication
    a[0::2, 1:-1:2] = (a[0::2, 0:-2:2] + a[0::2, 2::2]) // 2
    a[0::2, -1] = a[0::2, -2]
    # vertical pass: all odd rows, last row by replication
    a[1:-1:2, :] = (a[0:-2:2, :] + a[2::2, :]) // 2
    a[-1, :] = a[-2, :]


def interpolate_blue(a: np.ndarray) -> None:
    """Blue is known at (odd row, odd col); first row/column filled by replication."""
    a[1::2, 2::2] = (a[1::2, 1:-2:2] + a[1::2, 3::2]) // 2
    a[1::2, 0] = a[1::2, 1]
    a[2::2, :] = (a[1:-2:2, :] + a[3::2, :]) // 2
    a[0, :] = a[1, :]


def interpolate_green(a: np.ndarray) -> None:
    """Green is known where (row + col) is odd; missing pixels get the
    average of their four neighbours, mirrored at the borders."""
    h, w = a.shape
    padded = np.pad(a, 1, mode="reflect")
    avg = (padded[:-2, 1:-1] + padded[2:, 1:-1]
           + padded[1:-1, :-2] + padded[1:-1, 2:]) // 4
    missing = np.zeros((h, w), dtype=bool)
    missing[0::2, 0::2] = True
    missing[1::2, 1::2] = True
    a[missing] = avg[missing]


def main() -> None:
    filename = sys.argv[1] if len(sys.argv) > 1 else "raw.RAW"
    pixel_count = WIDTH * HEIGHT

    raw = np.fromfile(filename, dtype=np.uint8)
    expected = pixel_count * 5 // 4
    if raw.size != expected:
        sys.exit(f"Unexpected file size {raw.size} "
                 f"(expected {expected} for {WIDTH}x{HEIGHT} packed 10-bit)")

    bayer = unpack_10bit(raw, pixel_count).reshape(HEIGHT, WIDTH)
    red, green, blue = split_channels(bayer)

    start = time.perf_counter()
    interpolate_red(red)
    interpolate_green(green)
    interpolate_blue(blue)
    proc_time = time.perf_counter() - start
    print(f"Processing time (numpy): {proc_time * 1000:.3f} msec")

    # interleaved B,G,R 16-bit output, same layout as the C++ said.raw
    bgr = np.stack([blue, green, red], axis=-1)
    bgr.tofile("said_py.raw")

    # 8-bit preview image
    rgb8 = (np.stack([red, green, blue], axis=-1) >> 2).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(rgb8).save("result_py.png")
        print("Saved result_py.png")
    except ImportError:
        print("Pillow not installed - skipped PNG preview")

    try:
        import cv2
        cv2.imshow("result", (bgr * 64).astype(np.uint16))
        cv2.waitKey(0)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
