"""不确定性引导的深度监督。 / Uncertainty-guided depth supervision."""
import torch
import torch.nn.functional as F


def uncertainty_depth_loss(prediction, prior, uncertainty, valid_mask=None,
                           kind="huber", delta=0.02, reduction="active_mean"):
    """区间内不监督；区间外惩罚完整残差。 / Gate the full residual outside the interval.

    active_mean 按激活像素数平均；mean 使用有效像素数，sum 对应求和形式。
    active_mean divides by active pixels; mean divides by valid pixels; sum is unnormalized.
    """
    if prediction.shape != prior.shape or prediction.shape != uncertainty.shape:
        raise ValueError("Depth prediction, prior and uncertainty must have identical shapes")
    if kind not in ("huber", "l1") or reduction not in ("active_mean", "mean", "sum"):
        raise ValueError("Unsupported depth loss or reduction")
    if delta <= 0:
        raise ValueError("Huber delta must be positive")
    valid = torch.isfinite(prior) & torch.isfinite(uncertainty) & (uncertainty >= 0)
    if valid_mask is not None:
        if valid_mask.shape != prediction.shape:
            raise ValueError("Depth mask must have the same shape as depth")
        valid = valid & valid_mask.bool()
    # 不掩盖网络数值错误；只过滤无效监督。 / Do not silently hide invalid predictions.
    if not torch.isfinite(prediction).all():
        raise ValueError("Non-finite predicted depth")
    active = valid & ((prediction - prior).abs() > uncertainty)
    # 空集合返回有计算图的零，避免 mean(empty) 的 NaN。
    # Keep a differentiable zero when every ray lies inside its confidence interval.
    if not active.any():
        return prediction.sum() * 0.0
    if kind == "huber":
        errors = F.huber_loss(prediction[active], prior[active], delta=delta, reduction="sum")
    else:
        errors = (prediction[active] - prior[active]).abs().sum()
    if reduction == "active_mean":
        return errors / active.sum()
    if reduction == "mean":
        return errors / valid.sum()
    return errors
