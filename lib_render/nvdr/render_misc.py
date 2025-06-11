import torch


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.sqrt(
        torch.clamp(dot(x, x), min=eps)
    )  # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def scale_img_hwc(x: torch.Tensor, size, mag="bilinear", min="area") -> torch.Tensor:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]


def scale_img_nhwc(x: torch.Tensor, size, mag="bilinear", min="area") -> torch.Tensor:
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (
        x.shape[1] < size[0] and x.shape[2] < size[1]
    ), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if (
        x.shape[1] > size[0] and x.shape[2] > size[1]
    ):  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:  # Magnification
        if mag == "bilinear" or mag == "bicubic":
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


def avg_pool_nhwc(x: torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC
