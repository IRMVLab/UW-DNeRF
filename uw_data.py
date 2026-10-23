"""离线深度/不确定性读取与时间划分。 / Offline priors and temporal splits."""
from pathlib import Path
import numpy as np
import imageio.v2 as imageio


def load_scalar_maps(basedir, subdir, image_shape, scale=1.0):
    """按文件名排序与 RGB 对齐；禁止用深度冒充不确定性。
    Align sorted maps with sorted RGB frames; never substitute depth for uncertainty.
    image_shape: (frames, height, width). Multi-channel EXR uses its first channel.
    """
    directory = Path(basedir) / subdir
    if not directory.is_dir():
        raise FileNotFoundError("Required prior directory does not exist: {}".format(directory))
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".npy"}
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in suffixes)
    n, h, w = image_shape
    if len(files) != n:
        raise ValueError("{}: expected {} maps, found {}".format(directory, n, len(files)))
    maps = []
    for path in files:
        array = np.load(str(path), allow_pickle=False) if path.suffix.lower() == ".npy" else imageio.imread(str(path))
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        if array.ndim != 2 or not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("Expected finite, nonnegative scalar map: {}".format(path))
        if array.shape != (h, w):
            # 只允许同宽高比缩放，数值保持浮点精度。 / Resize without quantizing values.
            if abs(array.shape[0] / array.shape[1] - h / w) > 0.01:
                raise ValueError("Prior/RGB aspect ratio mismatch: {}".format(path))
            import cv2
            array = cv2.resize(array, (w, h), interpolation=cv2.INTER_LINEAR)
        maps.append(array * scale)
    return np.stack(maps).astype(np.float32)


def normalize_uncertainty(maps, lower=1e-5, upper=1e-4, mode="minmax"):
    """归一化到指定尺度，并处理恒定图。 / Normalize to the requested range, handling constants.

    raw 模式用于已标定到深度单位的输入。 / raw keeps calibrated depth-unit intervals.
    恒定图取下界，此时没有可用于排序的不确定性变化。
    A constant map uses the lower bound because no relative variation is available.
    """
    if mode not in ("raw", "minmax") or not 0 <= lower <= upper:
        raise ValueError("Invalid uncertainty normalization settings")
    maps = np.asarray(maps, dtype=np.float32)
    if not np.isfinite(maps).all() or (maps < 0).any():
        raise ValueError("Uncertainty must be finite and nonnegative")
    if mode == "raw":
        return maps.copy()
    span = float(maps.max()) - float(maps.min())
    if span <= 0:
        return np.full_like(maps, lower)
    return lower + (upper - lower) * ((maps - maps.min()) / span)


def temporal_split(n_frames, interval=8, offset=1, skip_frames=()):
    """offset=1 使用默认划分；offset=7 对齐论文每八帧留一帧。
    offset=1 is the default; offset=7 follows the paper's eighth-frame holdout.
    """
    if interval < 2 or not 0 <= offset < interval:
        raise ValueError("Split interval must be >=2 and 0 <= offset < interval")
    test = np.arange(offset, n_frames, interval, dtype=np.int64)
    if offset == 1:
        test = test[test < n_frames - 1]  # 默认保留末帧训练 / Keep the last frame for training.
    excluded = set(test.tolist()) | set(skip_frames)
    train = np.array([i for i in range(n_frames) if i not in excluded], dtype=np.int64)
    if len(train) == 0:
        raise ValueError("No training frames remain")
    return train, test
