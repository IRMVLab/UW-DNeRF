"""UW-DNeRF 深度评估指标。 / Depth evaluation metrics for UW-DNeRF."""
import numpy as np


def depth_metrics(prediction, reference, valid_mask=None, align_median=False):
    """默认保留绝对尺度；中位数对齐必须显式开启。
    Preserve absolute scale by default; median alignment must be requested explicitly.
    """
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape:
        raise ValueError('Predicted and reference depths must have identical shapes')
    valid = np.isfinite(reference) & (reference > 0)
    if valid_mask is not None:
        if np.shape(valid_mask) != reference.shape:
            raise ValueError('Depth mask shape mismatch')
        valid &= np.asarray(valid_mask, dtype=bool)
    if not valid.any():
        raise ValueError('No valid reference depth pixels')
    pred, gt = prediction[valid].astype(np.float64), reference[valid].astype(np.float64)
    if not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError('Invalid predicted depth in evaluated pixels')
    if align_median:
        if np.median(pred) <= 0:
            raise ValueError('Cannot median-align zero depth')
        pred = pred * (np.median(gt) / np.median(pred))
    error = pred - gt
    return dict(abs_rel=float(np.mean(np.abs(error) / gt)),
                sq_rel=float(np.mean(error ** 2 / gt)),
                rmse=float(np.sqrt(np.mean(error ** 2))))
