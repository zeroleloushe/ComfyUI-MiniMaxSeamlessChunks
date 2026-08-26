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

class MMH3_LatentChunkSplitter:
    """
    Splits a MiniMax H3 AV LATENT (NestedTensor video+audio, or a plain
    5D video tensor) into 2-5 overlapping chunks on the H3 keyframe grid.

    overlap_frames is in LATENT TOKENS. Default 5 = one 17-frame block.
    Every chunk starts at token index % 5 == 0 and is padded to 5k+2 so
    it is a valid standalone clip for the causal VisualVAE / DiT.

    Do NOT VAEDecode the chunks separately if you can avoid it — merge
    first, then one VAEDecode. If you must decode a chunk on its own,
    it will only look right when align_h3_grid is on (the old default
    overlap=2 started chunk 2 at phase 1/2/3/4 and pulsed).
    """

    DESCRIPTION = (
        "Режет MiniMax H3 AV-LATENT (NestedTensor видео+аудио) на 2-5 кусков "
        "по keyframe-сетке 5k+2. Каждый кусок начинается с keyframe "
        "(индекс % 5 == 0) и паддится до 5k+2 — иначе VAEDecode со второго "
        "куска даёт 17-кадровый пульс/мерцание. plan довести до Latent Chunk Merge. "
        "Декодировать лучше ОДИН раз после Merge, не каждый кусок по отдельности."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": (
                        "H3 AV-LATENT целиком (NestedTensor video[B,24,T,H,W] + "
                        "audio[B,32,2,Ta], либо уже распакованное видео 5D). "
                        "Обычный 4D image-latent не годится."
                    )}),
                "num_chunks": ("INT", {
                    "default": 2, "min": 2, "max": 5,
                    "tooltip": "На сколько кусков резать (2-5)."}),
                "overlap_frames": ("INT", {
                    "default": 5, "min": 0, "max": 40, "step": 5,
                    "tooltip": (
                        "Нахлёст в ЛАТЕНТ-токенах. Сетка H3 — кратно 5 "
                        "(1 блок = 17 пиксельных кадров ≈ 0.7с @24fps). "
                        "Значение будет округлено вниз до кратного 5. "
                        "Старый дефолт 2 как раз и давал мерцание: кусок 2 "
                        "начинался не с keyframe."
                    )}),
                "align_h3_grid": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "ON (рекомендуется): каждый кусок стартует с keyframe "
                        "(token % 5 == 0), overlap кратен 5, длина паддится до 5k+2. "
                        "OFF: старое поведение (равномерный сплит) — будет мерцать "
                        "на VAEDecode со второго куска."
                    )}),
                "pad_multiple": ("INT", {
                    "default": 5, "min": 0, "max": 32,
                    "tooltip": (
                        "Формула длины куска: tokens = pad_multiple*n + pad_remainder. "
                        "Для H3 это 5n+2. При align_h3_grid=ON это применяется "
                        "поверх сетки. 0 = не паддить."
                    )}),
                "pad_remainder": ("INT", {
                    "default": 2, "min": 0, "max": 32,
                    "tooltip": "Остаток формулы. Для H3 = 2 (длина 5k+2)."}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "LATENT", "LATENT", "STRING", "STRING")
    RETURN_NAMES = ("latent_chunk_1", "latent_chunk_2", "latent_chunk_3", "latent_chunk_4", "latent_chunk_5", "plan", "info")
    OUTPUT_TOOLTIPS = (
        "Латент-кусок 1 (NestedTensor, стартует с keyframe). Есть всегда.",
        "Латент-кусок 2. Есть, если num_chunks >= 2.",
        "Латент-кусок 3. Есть, если num_chunks >= 3.",
        "Латент-кусок 4. Есть, если num_chunks >= 4.",
        "Латент-кусок 5. Есть, только если num_chunks = 5.",
        "JSON-карта разбивки (в latent tokens) — довести до MMH3 Latent Chunk Merge. Подходит и для Audio Chunk Splitter.",
        "Отчёт: NestedTensor ли вход, T, 5k+2, фаза каждого куска, паддинг.",
    )
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Seamless Chunks"

    def run(self, latent, num_chunks, overlap_frames, align_h3_grid=True,
            pad_multiple=5, pad_remainder=2):
        samples = latent["samples"]
        video, audio, was_nested = unbind_samples(samples)
        video = validate_video(video, where="MMH3 Latent Chunk Splitter")
        total_frames = int(video.shape[2])

        if align_h3_grid:
            chunks = plan_chunks_h3(
                total_frames, num_chunks, overlap_frames,
                pad_multiple, pad_remainder,
            )
        else:
            chunks = plan_chunks(
                total_frames, num_chunks, overlap_frames,
                pad_multiple, pad_remainder,
            )

        dropped_noise_mask = "noise_mask" in latent and latent["noise_mask"] is not None

        notes = []
        if was_nested:
            notes.append(
                f"unbound NestedTensor: video {tuple(int(x) for x in video.shape)}"
                + (f" audio {tuple(int(x) for x in audio.shape)}" if audio is not None else " audio=None")
            )
        else:
            notes.append(
                f"plain 5D video latent {tuple(int(x) for x in video.shape)} "
                "(no audio member — output chunks will be plain too)"
            )
        notes.append(
            f"source T={total_frames}  5k+2={'yes' if is_h3_token_grid(total_frames) else 'NO'}  "
            f"pixel_frames≈{frames_for_tokens(total_frames)}  "
            f"align_h3_grid={'on' if align_h3_grid else 'OFF'}"
        )
        if not is_h3_token_grid(total_frames):
            notes.append(
                "WARNING: source T is not 5k+2. Even a single VAEDecode of the "
                "full latent may pulse. Encode/generate on the 17k+5 pixel grid."
            )

        outputs = []
        for c in chunks:
            phase = c.get("start_phase", c["raw_start"] % 5 if c["raw_start"] >= 0 else c["raw_start"])
            if align_h3_grid and phase != 0 and c["raw_start"] >= 0:
                notes.append(
                    f"chunk {c['index']}: start token {c['raw_start']} phase={phase} "
                    "(expected 0). This chunk will flicker if VAEDecoded alone."
                )
            elif c["raw_start"] >= 0:
                tag = "keyframe OK" if phase == 0 else f"PHASE {phase} — will flicker if decoded alone"
                notes.append(
                    f"chunk {c['index']}: tokens [{c['raw_start']}:{c['raw_end']}) "
                    f"len={c['raw_len']} pad→{c['final_len']}  {tag}"
                )
            v, a = slice_latent_chunk(video, audio, c, notes)
            outputs.append(pack_latent(v, a, was_nested, template=latent))

        while len(outputs) < 5:
            outputs.append(pack_latent(
                video[:, :, :1],
                audio[..., :1] if audio is not None else None,
                was_nested, template=latent,
            ))

        if dropped_noise_mask:
            notes.append(
                "исходный latent содержал noise_mask — он НЕ был разбит и не перенесён "
                "в куски (если второй проход требует noise_mask, добавьте его отдельно)"
            )

        notes.append(
            "DECODE: не декодируйте куски по отдельности — склейте Latent Chunk Merge "
            "и один VAEDecode на весь ролик. Отдельный VAEDecode куска 2+ без keyframe-"
            "выравнивания даёт 17-кадровое мерцание (это и был баг)."
        )

        plan_payload = {
            "total_frames": total_frames,
            "num_chunks": num_chunks,
            "overlap": overlap_frames,
            "overlap_snapped": chunks[0]["right_ov"] if len(chunks) > 1 else 0,
            "pad_multiple": pad_multiple,
            "pad_remainder": pad_remainder,
            "unit": "latent_frames",
            "align_h3_grid": bool(align_h3_grid),
            "was_nested": bool(was_nested),
            "source_on_grid": is_h3_token_grid(total_frames),
            "pixel_frames_est": frames_for_tokens(total_frames),
            "chunks": chunks,
        }
        plan_json = json.dumps(plan_payload)
        info = (
            f"total_latent_tokens={total_frames} num_chunks={num_chunks} "
            f"overlap={overlap_frames} align_h3_grid={align_h3_grid}\n"
            + "\n".join(notes)
        )

        return (outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], plan_json, info)
