"""
Unpack / slice / repack MiniMax H3 AV latents.

H3 does not store a plain 5D video tensor in `latent["samples"]`. The sampler
emits a NestedTensor pair:

    video  [B, 24, T_v, H/16, W/16]     time axis = dim 2
    audio  [B, 32, 2,  T_a]             time axis = dim 3

Naive `samples[:, :, start:end]` applies the SAME slice to both members.
Audio dim 2 is stereo (size 2), so chunk 1 accidentally keeps the full stereo
pair (slice 0:T clips to 0:2) while chunk 2 slices past it and comes out
empty or truncated — and the video slice of chunk 2 starts off the keyframe
grid. That is the flicker the splitter was producing even before merge.

This module unbinds, slices each stream on its real time axis, and rebuilds
the NestedTensor with ComfyUI's constructor.
"""

from __future__ import annotations

from typing import Any, Optional

from .plan import (
    FRAME_RESCALE,
    audio_token_range,
    frames_for_tokens,
    is_h3_token_grid,
)


def _try_nested_ctor():
    try:
        from comfy.nested_tensor import NestedTensor  # type: ignore
        return NestedTensor
    except Exception:
        return None


class _FallbackNested:
    """Duck-typed NestedTensor for tests / environments without ComfyUI."""

    is_nested = True

    def __init__(self, tensors):
        self.tensors = list(tensors)

    def unbind(self):
        return tuple(self.tensors)

    def dim(self):
        return self.tensors[0].dim() if self.tensors else 0

    @property
    def shape(self):
        return self.tensors[0].shape if self.tensors else ()


def is_nested_samples(samples: Any) -> bool:
    if samples is None:
        return False
    if getattr(samples, "is_nested", False):
        return True
    if hasattr(samples, "unbind") and hasattr(samples, "tensors"):
        return True
    return False


def unbind_samples(samples: Any):
    """Return (video, audio_or_None, was_nested).

    Accepts:
      - ComfyUI NestedTensor((video, audio))
      - a 5D plain video tensor [B, C, T, H, W]
      - a (video, audio) tuple/list
    """
    if samples is None:
        raise ValueError("latent['samples'] is None")

    if is_nested_samples(samples):
        parts = list(samples.unbind())
        if not parts:
            raise ValueError("NestedTensor has no members")
        video = parts[0]
        audio = parts[1] if len(parts) > 1 else None
        return video, audio, True

    if isinstance(samples, (tuple, list)):
        if not samples:
            raise ValueError("empty samples tuple")
        video = samples[0]
        audio = samples[1] if len(samples) > 1 else None
        return video, audio, True

    return samples, None, False


def bind_samples(video, audio, was_nested: bool):
    if was_nested and audio is not None:
        ctor = _try_nested_ctor()
        if ctor is not None:
            try:
                return ctor((video, audio))
            except TypeError:
                try:
                    return ctor([video, audio])
                except Exception:
                    pass
            except Exception:
                pass
        return _FallbackNested((video, audio))
    return video


def validate_video(video, *, where: str = "latent"):
    if video is None:
        raise ValueError(f"{where}: missing video tensor")
    if getattr(video, "dim", None) is None or video.dim() != 5:
        shape = tuple(video.shape) if hasattr(video, "shape") else type(video)
        raise ValueError(
            f"{where}: expected a 5D video latent [B, C, T, H, W], got shape {shape}. "
            "If this is a NestedTensor, it should have been unbound first."
        )
    return video


def describe_latent(latent: dict) -> str:
    """Human-readable report used by MMH3 Latent Info."""
    samples = latent.get("samples") if isinstance(latent, dict) else None
    lines = []
    try:
        video, audio, nested = unbind_samples(samples)
    except Exception as e:
        return f"FAILED to unbind: {e}\nsamples type={type(samples)}"

    lines.append(f"nested={'yes' if nested else 'no (plain tensor)'}")
    v = validate_video(video, where="video")
    b, c, t, h, w = (int(x) for x in v.shape)
    lines.append(f"video  [{b}, {c}, {t}, {h}, {w}]   (time=dim2, T={t})")
    if c != 24:
        lines.append(f"WARNING: H3 video is 24-channel, got C={c}")
    on_grid = is_h3_token_grid(t)
    phase = t % 5
    lines.append(
        f"grid   T={t}  5k+2={'YES' if on_grid else 'NO'}  "
        f"T%5={phase}  pixel_frames≈{frames_for_tokens(t)}"
    )
    if not on_grid:
        lines.append(
            "WARNING: T is not 5k+2. VAEDecode of this latent will run the "
            "causal 17-frame chunker off-phase — expect flicker/pulse."
        )
    if audio is not None:
        ash = tuple(int(x) for x in audio.shape)
        lines.append(f"audio  {list(ash)}   (time=dim3, Ta={ash[-1] if ash else '?'})")
        if len(ash) != 4:
            lines.append(f"WARNING: H3 audio is 4D [B,32,2,Ta], got ndim={len(ash)}")
        elif ash[1] != 32:
            lines.append(f"WARNING: H3 audio is 32-channel, got C={ash[1]}")
        expected_ta = int(round(frames_for_tokens(t) * FRAME_RESCALE))
        if abs(ash[-1] - expected_ta) > 2:
            lines.append(
                f"WARNING: audio Ta={ash[-1]} vs expected ≈{expected_ta} "
                f"(frames*{FRAME_RESCALE:.3f})"
            )
    else:
        lines.append("audio  (none) — video-only latent")

    if isinstance(latent, dict):
        extra = [k for k in latent.keys() if k != "samples"]
        if extra:
            lines.append("extra_keys " + ", ".join(extra))
    return "\n".join(lines)


def slice_video(video, t0: int, t1: int):
    """video [B,C,T,H,W] → [:, :, t0:t1]."""
    return video[:, :, t0:t1]


def slice_audio(audio, a0: int, a1: int):
    """audio [B,C,2,Ta] → [..., a0:a1] on the last dim."""
    if audio is None:
        return None
    return audio[..., a0:a1]


def cat_video(pieces: list):
    import torch
    return torch.cat(pieces, dim=2)


def cat_audio(pieces: list):
    import torch
    return torch.cat(pieces, dim=-1)


def pad_video_tail(video, n: int):
    if n <= 0:
        return video
    tail = video[:, :, -1:].repeat(1, 1, n, 1, 1)
    import torch
    return torch.cat([video, tail], dim=2)


def pad_video_head(video, n: int):
    if n <= 0:
        return video
    head = video[:, :, :1].repeat(1, 1, n, 1, 1)
    import torch
    return torch.cat([head, video], dim=2)


def pad_audio_tail(audio, n: int):
    if audio is None or n <= 0:
        return audio
    tail = audio[..., -1:].repeat(*([1] * (audio.dim() - 1)), n)
    import torch
    return torch.cat([audio, tail], dim=-1)


def pad_audio_head(audio, n: int):
    if audio is None or n <= 0:
        return audio
    head = audio[..., :1].repeat(*([1] * (audio.dim() - 1)), n)
    import torch
    return torch.cat([head, audio], dim=-1)


def slice_av_window(video, audio, t0: int, t1: int, total_audio: Optional[int] = None):
    """Slice both streams for video-token window [t0, t1).

    t0/t1 may be outside [0, T); caller is expected to clamp + pad afterwards.
    Audio range is derived from pixel-frame coverage of the *clamped* token span.
    """
    T = int(video.shape[2])
    t0c = max(0, t0)
    t1c = min(T, t1)
    v = slice_video(video, t0c, t1c)
    a = None
    a0 = a1 = 0
    if audio is not None:
        Ta = int(audio.shape[-1])
        tot = Ta if total_audio is None else total_audio
        a0, a1 = audio_token_range(t0c, t1c, tot)
        a = slice_audio(audio, a0, a1)
    return v, a, t0c, t1c, a0, a1


def pack_latent(video, audio, was_nested: bool, template: Optional[dict] = None,
                noise_mask=None) -> dict:
    out = {}
    if template:
        for k, val in template.items():
            if k in ("samples", "noise_mask"):
                continue
            out[k] = val
    out["samples"] = bind_samples(video, audio, was_nested)
    if noise_mask is not None:
        out["noise_mask"] = noise_mask
    return out
