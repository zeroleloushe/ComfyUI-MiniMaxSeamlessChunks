import json
import torch
import numpy as np

from .plan import (
    frames_for_tokens,
    is_h3_token_grid,
    minimax_h3_frame_count,
    plan_chunks,
    plan_chunks_h3,
)
from .av_latent import (
    cat_audio,
    cat_video,
    describe_latent,
    pack_latent,
    pad_audio_head,
    pad_audio_tail,
    pad_video_head,
    pad_video_tail,
    slice_av_window,
    unbind_samples,
    validate_video,
    slice_latent_chunk,
)

def _latent_crossfade(a, b, mode):
    """a, b: [B,C,T,H,W] tensors, equal T along dim=2."""
    t_len = a.shape[2]
    if t_len == 0:
        return a
    device, dtype = a.device, a.dtype
    t = torch.linspace(0.0, 1.0, t_len, device=device, dtype=dtype).view(1, 1, t_len, 1, 1)
    if mode == "smoothstep":
        t = t * t * (3 - 2 * t)
        return a * (1 - t) + b * t
    if mode == "equal_energy":
        alpha = torch.sqrt(t)
        beta = torch.sqrt(1 - t)
        return a * beta + b * alpha
    if mode == "hard_cut":
        mid = t_len // 2
        return torch.cat([a[:, :, :mid], b[:, :, mid:]], dim=2)
    if mode == "causal":
        # Keep the already-generated prefix (a); drop the warmup prefix of b.
        # For unprocessed chunks this is identity. For independently sampled
        # chunks this prefers the earlier causal continuation over a blend of
        # two different VAE phases — which is what used to flicker.
        return a
    return a * (1 - t) + b * t  # linear


def _audio_latent_crossfade(a, b, mode):
    """a, b: [B,C,2,Ta] equal Ta along dim=-1."""
    n = a.shape[-1]
    if n == 0:
        return a
    if mode == "causal":
        return a
    device, dtype = a.device, a.dtype
    t = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    view = [1] * a.dim()
    view[-1] = n
    t = t.view(*view)
    if mode == "smoothstep":
        t = t * t * (3 - 2 * t)
        return a * (1 - t) + b * t
    if mode == "equal_energy":
        return a * torch.sqrt(1 - t) + b * torch.sqrt(t)
    if mode == "hard_cut":
        mid = n // 2
        return torch.cat([a[..., :mid], b[..., mid:]], dim=-1)
    return a * (1 - t) + b * t


class MMH3_LatentChunkMerge:
    """
    Reassembles processed H3 AV LATENT chunks back into one NestedTensor
    whose video T matches the source exactly. Decode ONCE afterwards.

    blend_mode `causal` (default) keeps the earlier chunk's overlap and
    drops the later chunk's warmup prefix — identity on unprocessed
    splits, and the only mode that does not average two different VAE
    phases after an independent 2nd pass.
    """

    DESCRIPTION = (
        "Собирает обработанные латент-куски обратно в NestedTensor точной "
        "исходной длины. blend_mode=causal (по умолчанию) не смешивает две "
        "разные фазы VAE на стыке — это и убирает мерцание. Декодируйте "
        "результат ОДНИМ VAEDecode."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("STRING", {
                    "forceInput": True,
                    "tooltip": "plan из MMH3 Latent Chunk Splitter. Обязателен."}),
                "latent_chunk_1": ("LATENT", {
                    "tooltip": "Обработанный латент-кусок 1. Обязателен всегда."}),
                "blend_mode": (["causal", "smoothstep", "linear", "equal_energy", "hard_cut"], {
                    "default": "causal",
                    "tooltip": (
                        "Как стыковать нахлёст.\n"
                        "causal — оставить хвост РАННЕГО куска, отбросить warmup "
                        "позднего (по умолчанию, нет смешения двух фаз VAE).\n"
                        "smoothstep / linear / equal_energy — кроссфейд в латенте "
                        "(имеет смысл только если оба куска прогнаны одним и тем же "
                        "2-м проходом на выровненной сетке).\n"
                        "hard_cut — разрез посередине нахлёста."
                    )}),
            },
            "optional": {
                "latent_chunk_2": ("LATENT", {"tooltip": "Обработанный латент-кусок 2. Нужен, если было num_chunks >= 2."}),
                "latent_chunk_3": ("LATENT", {"tooltip": "Обработанный латент-кусок 3. Нужен, если было num_chunks >= 3."}),
                "latent_chunk_4": ("LATENT", {"tooltip": "Обработанный латент-кусок 4. Нужен, если было num_chunks >= 4."}),
                "latent_chunk_5": ("LATENT", {"tooltip": "Обработанный латент-кусок 5. Нужен только если было num_chunks = 5."}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    OUTPUT_TOOLTIPS = (
        "Готовый склеенный AV-LATENT (NestedTensor если вход был nested). Один VAEDecode.",
        "Текстовый отчёт о склейке: T, 5k+2, nested, расхождения.",
    )
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Seamless Chunks"

    def run(self, plan, latent_chunk_1, blend_mode, latent_chunk_2=None,
            latent_chunk_3=None, latent_chunk_4=None, latent_chunk_5=None):
        payload = json.loads(plan)
        chunks_meta = payload["chunks"]
        num_chunks = payload["num_chunks"]
        total_frames = payload["total_frames"]
        was_nested_plan = payload.get("was_nested", None)

        raw_inputs = [latent_chunk_1, latent_chunk_2, latent_chunk_3, latent_chunk_4, latent_chunk_5][:num_chunks]
        for i, t in enumerate(raw_inputs):
            if t is None:
                raise ValueError(f"latent_chunk_{i+1} is required (plan expects {num_chunks} chunks) but was not connected")

        unbound = []
        was_nested = False
        for i, c in enumerate(raw_inputs):
            v, a, nested = unbind_samples(c["samples"])
            v = validate_video(v, where=f"latent_chunk_{i+1}")
            unbound.append((v, a, nested, c))
            was_nested = was_nested or nested

        v0, a0, _, tmpl = unbound[0]
        b0, ch0, h0, w0 = int(v0.shape[0]), int(v0.shape[1]), int(v0.shape[3]), int(v0.shape[4])
        for i, (v, a, nested, _) in enumerate(unbound):
            if (int(v.shape[0]) != b0 or int(v.shape[1]) != ch0
                    or int(v.shape[3]) != h0 or int(v.shape[4]) != w0):
                raise ValueError(
                    f"latent_chunk_{i+1} video shape {tuple(int(x) for x in v.shape)} does not match "
                    f"latent_chunk_1's B/C/H/W {(b0, ch0, h0, w0)}. "
                    f"All chunks must have gone through the same spatial upscale."
                )

        if was_nested_plan is False and was_nested:
            pass

        notes = []
        stripped_v = []
        stripped_a = []
        for meta, (v, a, nested, _) in zip(chunks_meta, unbound):
            expected_final = meta["final_len"]
            if int(v.shape[2]) != expected_final:
                notes.append(
                    f"chunk {meta['index']}: expected {expected_final} latent tokens after "
                    f"processing, got {int(v.shape[2])} (sampler changed length?)"
                )
            raw_len = meta["raw_len"]
            piece_v = v[:, :, :raw_len]
            if int(piece_v.shape[2]) < raw_len:
                piece_v = pad_video_tail(piece_v, raw_len - int(piece_v.shape[2]))
                notes.append(
                    f"chunk {meta['index']}: padded video to restore expected raw_len={raw_len}"
                )
            stripped_v.append(piece_v)

            if a is not None:
                stripped_a.append(a)
            else:
                stripped_a.append(None)

        result_v = []
        result_a = []
        have_audio = all(x is not None for x in stripped_a)

        for i, meta in enumerate(chunks_meta):
            piece_v = stripped_v[i]
            left_ov = meta["left_ov"]
            right_ov = meta["right_ov"]
            core_len = meta["core_len"]
            core_v = piece_v[:, :, left_ov: left_ov + core_len]

            if right_ov > 0:
                keep_len = core_len - right_ov
                result_v.append(core_v[:, :, :keep_len])
                this_tail = core_v[:, :, keep_len:]
                next_left_ctx = stripped_v[i + 1][:, :, :right_ov]
                if this_tail.shape[2] != next_left_ctx.shape[2]:
                    n = min(int(this_tail.shape[2]), int(next_left_ctx.shape[2]))
                    this_tail, next_left_ctx = this_tail[:, :, -n:], next_left_ctx[:, :, :n]
                result_v.append(_latent_crossfade(this_tail, next_left_ctx, blend_mode))
            else:
                result_v.append(core_v)

            if have_audio:
                piece_a = stripped_a[i]
                Ta = int(piece_a.shape[-1])
                raw_len_v = meta["raw_len"]
                def _a_span(tok0, tok1):
                    p0 = tok0 / max(raw_len_v, 1)
                    p1 = tok1 / max(raw_len_v, 1)
                    return int(round(p0 * Ta)), int(round(p1 * Ta))

                a_left0, a_core0 = _a_span(0, left_ov)
                a_core1 = _a_span(left_ov, left_ov + core_len)[1]
                core_a = piece_a[..., a_core0:a_core1]
                if right_ov > 0:
                    keep_tok = left_ov + (core_len - right_ov)
                    a_keep = _a_span(left_ov, keep_tok)[1] - a_core0
                    a_keep = max(0, min(int(core_a.shape[-1]), a_keep))
                    result_a.append(core_a[..., :a_keep])
                    this_tail_a = core_a[..., a_keep:]
                    n_next = _a_span(0, right_ov)[1]
                    next_left_a = stripped_a[i + 1][..., :n_next]
                    if this_tail_a.shape[-1] != next_left_a.shape[-1]:
                        n = min(int(this_tail_a.shape[-1]), int(next_left_a.shape[-1]))
                        this_tail_a, next_left_a = this_tail_a[..., -n:], next_left_a[..., :n]
                    if n > 0:
                        result_a.append(_audio_latent_crossfade(this_tail_a, next_left_a, blend_mode))
                else:
                    result_a.append(core_a)

        final_v = cat_video(result_v)
        final_a = cat_audio(result_a) if have_audio and result_a else None

        if int(final_v.shape[2]) != total_frames:
            notes.append(
                f"WARNING: reconstructed video T={int(final_v.shape[2])} != "
                f"source total_frames {total_frames}"
            )
        if not is_h3_token_grid(int(final_v.shape[2])):
            notes.append(
                f"WARNING: merged T={int(final_v.shape[2])} is not 5k+2 — "
                "VAEDecode will phase-shift from here on (17-frame pulse)."
            )
        else:
            notes.append(f"merged T={int(final_v.shape[2])} is on 5k+2 grid, pixel_frames≈{frames_for_tokens(int(final_v.shape[2]))}")

        info = (
            f"merged {num_chunks} latent chunks -> T={int(final_v.shape[2])} "
            f"(source {total_frames}), blend_mode={blend_mode}, "
            f"nested={'yes' if was_nested else 'no'}\n"
            + ("\n".join(notes) if notes else "no issues")
        )

        return (pack_latent(final_v, final_a, was_nested, template=tmpl), info)


class MMH3_LatentInfo:
    """Inspect an H3 AV latent: NestedTensor?, shapes, T, 5k+2, phase, audio match."""

    DESCRIPTION = (
        "Диагностика H3-латента: NestedTensor или плоский тензор, shape видео/"
        "аудио, T, попадает ли в сетку 5k+2, оценка пиксельных кадров. "
        "Подключите к выходу Splitter'а или к исходнику, чтобы увидеть, "
        "почему мерцает."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Любой H3 (или нет) LATENT — на вход или выход Splitter/Merge/KSampler."}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("info", "latent_tokens", "pixel_frames_est")
    OUTPUT_TOOLTIPS = (
        "Текстовый отчёт: nested, shapes, 5k+2, предупреждения.",
        "T видео-латента (dim 2).",
        "Пиксельные кадры по сетке (1,4,4,4,4) для этого T.",
    )
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Seamless Chunks"

    def run(self, latent):
        info = describe_latent(latent)
        try:
            video, _, _ = unbind_samples(latent["samples"])
            t = int(video.shape[2])
            return (info, t, frames_for_tokens(t))
        except Exception:
            return (info, 0, 0)


class MMH3_LastFrames:
    """Extracts the last N frames of an IMAGE batch."""

    DESCRIPTION = (
        "Берёт последние N кадров ролика — удобно прокинуть тот же "
        "референс в следующую генерацию MiniMax H3."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Ролик (IMAGE-батч)."}),
                "count": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Сколько последних кадров взять."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("last_frames",)
    OUTPUT_TOOLTIPS = ("Последние N кадров исходного ролика.",)
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Seamless Chunks"

    def run(self, images, count):
        n = min(count, images.shape[0])
        return (images[-n:],)
