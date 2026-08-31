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
)


# --------------------------------------------------------------------------
# 1. Frame calculator — figure out the REAL frame count before/after
#    generation, so downstream nodes never guess.
# --------------------------------------------------------------------------
class MMH3_FrameCalculator:
    """
    Computes the frame count MiniMax H3 will actually render for a
    requested duration (17k+5 quantization @24fps), and the exact seconds
    that corresponds to. Feed `frames` into the splitter node, and
    `seconds` into your MiniMax generation node's duration field if you
    want the request to match a whole quantized step.
    """

    DESCRIPTION = (
        "Считает РЕАЛЬНОЕ число кадров, которое MiniMax H3 отдаст для "
        "запрошенной длительности (квантование 17k+5 @24fps: 10с -> 243 "
        "кадра, 15с -> 362 кадра). Поставьте эту ноду ДО генерации, чтобы "
        "знать точное число кадров заранее, и/или ПОСЛЕ, чтобы проверить, "
        "что скачанный ролик соответствует ожиданиям."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "requested_seconds": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 15.0, "step": 0.1,
                    "tooltip": "Желаемая длительность ролика в секундах (как вы бы ввели в поле duration генератора MiniMax H3)."}),
                "fps": ("INT", {
                    "default": 24, "min": 1, "max": 60,
                    "tooltip": "Частота кадров, с которой рендерит MiniMax H3. По умолчанию 24 — менять не нужно, если явно не знаете, что модель отдаёт другой fps."}),
                "block": ("INT", {
                    "default": 17, "min": 1, "max": 256,
                    "tooltip": "Размер блока квантования кадров MiniMax H3. Менять не нужно — оставлено настраиваемым на случай, если формула изменится в будущей версии модели."}),
                "remainder": ("INT", {
                    "default": 5, "min": 0, "max": 256,
                    "tooltip": "Остаток формулы кадров MiniMax H3 (frames = block*k + remainder). Менять не нужно."}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "INT")
    RETURN_NAMES = ("frames", "seconds", "fps")
    OUTPUT_TOOLTIPS = (
        "Точное число кадров, которое реально отдаст MiniMax H3. Подключите в num_chunks-логику или просто для справки.",
        "Точная длительность в секундах, соответствующая quantized frames. Можно подключить на вход duration генератора MiniMax H3.",
        "FPS, тот же, что подан на вход (проброшен для удобства подключения дальше по графу).",
    )
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Seamless Chunks"

    def run(self, requested_seconds, fps, block, remainder):
        frames = minimax_h3_frame_count(requested_seconds, fps, block, remainder)
        seconds = frames / float(fps)
        return (frames, seconds, fps)


# --------------------------------------------------------------------------
# 2. Chunk splitter — cuts the decoded MiniMax clip into up to 4
#    overlapping pieces ready to be fed into ANY external upscaler /
#    sampler graph. Outputs are plain IMAGE batches, so wire chunk_N into
#    whatever upscale/KSampler chain you want; the plan travels alongside
#    as a JSON string for the merge node.
# --------------------------------------------------------------------------
class MMH3_ChunkSplitter:
    """
    Splits an IMAGE batch (the decoded MiniMax H3 clip) into 2-5
    overlapping chunks for independent upscale + second-sampler-pass
    processing.

    - If the actual number of frames you feed in is shorter than expected
      near a boundary (decode/rounding mismatch), the missing frames are
      filled by duplicating the last available frame, and this is noted
      in `info`.
    - Each chunk is additionally padded (duplicated last frame) up to
      pad_multiple*k + pad_remainder frames so it's a valid input length
      for a second-pass video sampler with temporal compression (most
      latent video models want 4n+1 frames — that's the default).
    - Connect chunk_1..chunk_5 to your upscaler / KSampler chain
      (VAEEncode -> KSampler -> VAEDecode, or an RTX/TensorRT upscale
      node — any node chain that takes/returns IMAGE works). Only
      connect as many chunk_N outputs as `num_chunks`; the rest can stay
      unconnected.
    - If your second pass needs synchronized audio (e.g. a lipsync-aware
      model/conditioning), use MMH3 Audio Chunk Splitter with the same
      `plan` output — it cuts the audio at the exact same timeline
      positions as this node cuts the video, overlap included.
    """

    DESCRIPTION = (
        "Режет готовый декодированный ролик MiniMax H3 на 2-5 кусков с "
        "нахлёстом для независимого апскейла и второго прохода семплером. "
        "chunk_1..chunk_5 — обычный IMAGE, подключайте куда угодно "
        "(апскейлер, VAEEncode->KSampler->VAEDecode). plan обязательно "
        "довести до MMH3 Chunk Merge (и до MMH3 Audio Chunk Splitter, если "
        "нужен синхронный звук) — это единственная нить, которая "
        "гарантирует точную сборку обратно."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Декодированный ролик MiniMax H3 целиком (весь батч кадров), например выход VHS Load Video."}),
                "num_chunks": ("INT", {
                    "default": 4, "min": 2, "max": 5,
                    "tooltip": "На сколько кусков резать (2-5). Столько же выходов chunk_N нужно будет подключить дальше по графу."}),
                "overlap_frames": ("INT", {
                    "default": 8, "min": 0, "max": 64,
                    "tooltip": "Сколько кадров соседние куски делят между собой для кроссфейда на стыке. 8 кадров при 24fps ≈ треть секунды. Для быстрого движения в кадре ставьте 12-16."}),
                "pad_multiple": ("INT", {
                    "default": 4, "min": 0, "max": 64,
                    "tooltip": "Множитель формулы длины входа для вашего второго семплера (frames = pad_multiple*n + pad_remainder). 4 — типично для видео-моделей со сжатием времени x4. Поставьте 0, если семплеру всё равно, сколько кадров на входе."}),
                "pad_remainder": ("INT", {
                    "default": 1, "min": 0, "max": 64,
                    "tooltip": "Остаток той же формулы. Вместе с pad_multiple определяет, сколько кадров-дублей допишет нода в конец каждого куска перед семплером."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5", "plan", "info")
    OUTPUT_TOOLTIPS = (
        "Кусок 1: подключить в апскейлер/семплер. Есть всегда.",
        "Кусок 2: подключить в апскейлер/семплер. Есть, если num_chunks >= 2.",
        "Кусок 3: подключить в апскейлер/семплер. Есть, если num_chunks >= 3.",
        "Кусок 4: подключить в апскейлер/семплер. Есть, если num_chunks >= 4.",
        "Кусок 5: подключить в апскейлер/семплер. Есть, только если num_chunks = 5.",
        "JSON-карта разбивки — обязательно провести отдельным проводом в MMH3 Chunk Merge.plan (и в MMH3 Audio Chunk Splitter.plan, если нужен звук).",
        "Текстовый отчёт: что и где было задублировано при паддинге. Удобно вывести в Show Text / Preview Any для отладки.",
    )
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Seamless Chunks"

    def run(self, images, num_chunks, overlap_frames, pad_multiple, pad_remainder):
        total_frames = images.shape[0]
        chunks = plan_chunks(total_frames, num_chunks, overlap_frames,
                              pad_multiple, pad_remainder)

        notes = []
        outputs = []
        for c in chunks:
            start, end = c["raw_start"], c["raw_end"]
            start_clamped = max(0, start)
            end_clamped = min(total_frames, end)
            piece = images[start_clamped:end_clamped]

            missing_left = start_clamped - start
            missing_right = end - end_clamped
            if missing_left > 0:
                first = piece[:1].repeat(missing_left, 1, 1, 1)
                piece = torch.cat([first, piece], dim=0)
                notes.append(f"chunk {c['index']}: padded {missing_left} frame(s) at left edge (source too short)")
            if missing_right > 0:
                last = piece[-1:].repeat(missing_right, 1, 1, 1)
                piece = torch.cat([piece, last], dim=0)
                notes.append(f"chunk {c['index']}: padded {missing_right} frame(s) at right edge (source too short)")

            if c["pad"] > 0:
                last = piece[-1:].repeat(c["pad"], 1, 1, 1)
                piece = torch.cat([piece, last], dim=0)
                notes.append(f"chunk {c['index']}: +{c['pad']} duplicated trailing frame(s) "
                             f"to satisfy {pad_multiple}n+{pad_remainder} sampler formula "
                             f"({c['raw_len']} -> {c['final_len']})")

            outputs.append(piece)

        # pad the outputs list up to 5 slots with tiny dummy tensors so the
        # fixed-arity RETURN_TYPES is always satisfiable; unused sockets
        # simply won't be connected to anything downstream.
        while len(outputs) < 5:
            outputs.append(images[:1])

        plan_payload = {
            "total_frames": total_frames,
            "num_chunks": num_chunks,
            "overlap": overlap_frames,
            "pad_multiple": pad_multiple,
            "pad_remainder": pad_remainder,
            "chunks": chunks,
        }
        plan_json = json.dumps(plan_payload)
        info = f"total_frames={total_frames} num_chunks={num_chunks} overlap={overlap_frames}\n" + \
               ("\n".join(notes) if notes else "no padding needed")

        return (outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], plan_json, info)


# --------------------------------------------------------------------------
# 3. Seamless merge — strips the padding each chunk picked up, crossfades
#    the overlap regions, and concatenates back to EXACTLY total_frames.
# --------------------------------------------------------------------------
def _flow_align_numpy(a_np, b_np):
    """
    a_np, b_np: single frames, HxWx3 float32 in [0,1] (numpy, RGB).
    Warps b_np onto a_np's geometry using dense optical flow, to cancel
    out sub-pixel jitter between two independently-processed renders of
    the same nominal timestamp (not real motion — there is none between
    them, only render noise/drift). Returns warped b_np.
    """
    import cv2  # lazy import — only needed for blend_mode="flow_align"
    a8 = np.clip(a_np * 255.0, 0, 255).astype(np.uint8)
    b8 = np.clip(b_np * 255.0, 0, 255).astype(np.uint8)
    a_gray = cv2.cvtColor(a8, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b8, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        b_gray, a_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    h, w = a_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(b8, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    return warped.astype(np.float32) / 255.0


def _crossfade_flow_align(a, b):
    """
    EXPERIMENTAL. a, b: [T,H,W,C] torch tensors of equal T (same as
    _crossfade). For each frame pair, first warps b's frame onto a's
    frame to remove sub-pixel jitter/drift, THEN applies the same
    smoothstep schedule as the plain "smoothstep" mode. Falls back to
    plain smoothstep (no warp) per-frame if opencv-python is not
    installed or a frame fails to process, and reports that via the
    returned `used_fallback` flag.

    Measured on synthetic test content in this repo's test suite
    (moving textured pattern + simulated sub-pixel misregistration
    between the two "renders"), this did NOT outperform plain
    smoothstep — Farneback flow estimation on smooth/low-texture
    regions is itself noisy, and the warp/remap step adds its own
    interpolation error that can exceed what it corrects. Kept as an
    opt-in experimental mode in case it helps on specific real
    upscaler/sampler jitter patterns not captured by the synthetic
    tests; "smoothstep" remains the default and the generally safer
    choice.
    """
    device, dtype = a.device, a.dtype
    a_np = a.detach().cpu().numpy().astype(np.float32)
    b_np = b.detach().cpu().numpy().astype(np.float32)
    t_len = a_np.shape[0]

    used_fallback = False
    try:
        import cv2  # noqa: F401  (import check only, real import happens per-frame below)
    except ImportError:
        used_fallback = True

    warped_b = np.empty_like(b_np)
    if used_fallback:
        warped_b[:] = b_np
    else:
        for i in range(t_len):
            try:
                warped_b[i] = _flow_align_numpy(a_np[i], b_np[i])
            except Exception:
                # never let a single bad frame kill the whole merge —
                # just skip alignment for that frame and note it
                warped_b[i] = b_np[i]
                used_fallback = True

    a_t = torch.from_numpy(a_np).to(device=device, dtype=dtype)
    warped_b_t = torch.from_numpy(warped_b).to(device=device, dtype=dtype)

    t = torch.linspace(0.0, 1.0, t_len, device=device, dtype=dtype).view(t_len, 1, 1, 1)
    t = t * t * (3 - 2 * t)  # smoothstep schedule
    blended = a_t * (1 - t) + warped_b_t * t
    return blended, used_fallback


def _crossfade(a, b, mode):
    """a, b: [T,H,W,C] tensors of equal T. Returns (blended, used_fallback)."""
    t_len = a.shape[0]
    if t_len == 0:
        return a, False
    device = a.device
    dtype = a.dtype
    if mode == "flow_align":
        return _crossfade_flow_align(a, b)
    t = torch.linspace(0.0, 1.0, t_len, device=device, dtype=dtype).view(t_len, 1, 1, 1)
    if mode == "smoothstep":
        t = t * t * (3 - 2 * t)
        return a * (1 - t) + b * t, False
    if mode == "equal_energy":
        alpha = torch.sqrt(t)
        beta = torch.sqrt(1 - t)
        return a * beta + b * alpha, False
    if mode == "hard_cut":
        mid = t_len // 2
        return torch.cat([a[:mid], b[mid:]], dim=0), False
    # linear
    return a * (1 - t) + b * t, False


class MMH3_ChunkMerge:
    """
    Reassembles processed chunks (post-upscale, post second-sampler-pass)
    back into one clip that has EXACTLY the same frame count as the
    original MiniMax H3 render. Strips the padding the splitter added,
    then crossfades the shared overlap regions between neighbouring
    chunks instead of hard-cutting, so the seams aren't visible.

    Feed it the same `plan` string the splitter produced, and the
    processed chunk_1..chunk_5 IMAGE batches (must be same H/W as each
    other — resize before merging if your upscaler changed resolution
    inconsistently, e.g. if only some chunks were upscaled).
    """

    DESCRIPTION = (
        "Собирает обработанные (апскейл + второй проход семплером) куски "
        "обратно в один ролик. Длина на выходе всегда точно равна длине "
        "исходного ролика — если что-то не сошлось, это будет написано в "
        "info, а не тихо съедено."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("STRING", {
                    "forceInput": True,
                    "tooltip": "JSON-карта разбивки — тот самый provод из выхода plan ноды MMH3 Chunk Splitter. Обязателен."}),
                "chunk_1": ("IMAGE", {
                    "tooltip": "Обработанный (после апскейла/семплера) кусок 1. Обязателен всегда."}),
                "blend_mode": (["smoothstep", "flow_align", "linear", "equal_energy", "hard_cut"], {
                    "default": "smoothstep",
                    "tooltip": "Как смешивать зону нахлёста на стыке кусков.\nsmoothstep — плавный кроссфейд, мало мерцания (по умолчанию, рекомендуется).\nflow_align — ЭКСПЕРИМЕНТАЛЬНО: выравнивает микро-смещение между версиями кадра через optical flow (opencv) перед smoothstep. На синтетических тестах пакета выигрыша над обычным smoothstep не показал (оценка потока сама вносит шум) — используйте только если на вашей реальной паре апскейлер+семплер видите иначе, и сравнивайте на глаз. Требует opencv-python (cv2); без него тихо откатывается на smoothstep.\nlinear — простой кроссфейд без сглаживания краёв.\nequal_energy — сохраняет яркость при смешивании контрастных сцен.\nhard_cut — без смешивания, разрез посередине нахлёста."}),
            },
            "optional": {
                "chunk_2": ("IMAGE", {
                    "tooltip": "Обработанный кусок 2. Нужен, если в Splitter стояло num_chunks >= 2 (то есть почти всегда)."}),
                "chunk_3": ("IMAGE", {
                    "tooltip": "Обработанный кусок 3. Нужен, если в Splitter стояло num_chunks >= 3."}),
                "chunk_4": ("IMAGE", {
                    "tooltip": "Обработанный кусок 4. Нужен, если в Splitter стояло num_chunks >= 4."}),
                "chunk_5": ("IMAGE", {
                    "tooltip": "Обработанный кусок 5. Нужен только если в Splitter стояло num_chunks = 5."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    OUTPUT_TOOLTIPS = (
        "Готовый склеенный ролик, число кадров = числу кадров исходного ролика MiniMax H3. Подать в Save/Combine Video.",
        "Текстовый отчёт о склейке: сколько кадров получилось, были ли расхождения. Проверяйте, если видео выглядит не так.",
    )
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Seamless Chunks"

    def run(self, plan, chunk_1, blend_mode, chunk_2=None, chunk_3=None, chunk_4=None, chunk_5=None):
        payload = json.loads(plan)
        chunks_meta = payload["chunks"]
        num_chunks = payload["num_chunks"]
        total_frames = payload["total_frames"]

        raw_inputs = [chunk_1, chunk_2, chunk_3, chunk_4, chunk_5][:num_chunks]
        for i, t in enumerate(raw_inputs):
            if t is None:
                raise ValueError(f"chunk_{i+1} is required (plan expects {num_chunks} chunks) but was not connected")

        # sanity: resolution must match across chunks for concatenation
        h, w = raw_inputs[0].shape[1], raw_inputs[0].shape[2]
        for i, t in enumerate(raw_inputs):
            if t.shape[1] != h or t.shape[2] != w:
                raise ValueError(
                    f"chunk_{i+1} resolution {t.shape[1]}x{t.shape[2]} does not match "
                    f"chunk_1 resolution {h}x{w}. Resize all chunks to the same size "
                    f"before merging (e.g. run the same upscale factor on every chunk)."
                )

        stripped = []
        notes = []
        for meta, tensor in zip(chunks_meta, raw_inputs):
            expected_final = meta["final_len"]
            if tensor.shape[0] != expected_final:
                notes.append(
                    f"chunk {meta['index']}: expected {expected_final} frames after "
                    f"processing, got {tensor.shape[0]} (upscaler/sampler changed frame "
                    f"count?) — using min(available, raw_len)"
                )
            raw_len = meta["raw_len"]
            piece = tensor[:raw_len]  # drop the pad-multiple padding
            if piece.shape[0] < raw_len:
                # upscaler returned fewer frames than expected: pad by duplicating
                last = piece[-1:].repeat(raw_len - piece.shape[0], 1, 1, 1)
                piece = torch.cat([piece, last], dim=0)
                notes.append(f"chunk {meta['index']}: padded {raw_len - piece.shape[0]} "
                              f"frame(s) after processing to restore expected length")
            stripped.append(piece)

        # For each chunk, `core` is the slice of the chunk that corresponds
        # 1:1 to its reserved timeline positions (no neighbour overlap).
        # For every internal boundary (all but the last chunk), the last
        # `overlap` frames of this chunk's core are NOT emitted directly;
        # instead they are crossfaded against the next chunk's left-context
        # (which covers those same timeline positions, reprocessed inside
        # chunk i+1) and that single blended run of `overlap` frames is
        # emitted in their place. This keeps the total output length equal
        # to sum(core_len) == total_frames exactly, with no seam.
        result_pieces = []
        flow_fallback_boundaries = []
        for i, meta in enumerate(chunks_meta):
            piece = stripped[i]
            left_ov = meta["left_ov"]
            right_ov = meta["right_ov"]
            core_len = meta["core_len"]
            core = piece[left_ov: left_ov + core_len]

            if right_ov > 0:
                keep_len = core_len - right_ov
                result_pieces.append(core[:keep_len])
                this_tail = core[keep_len:]
                next_left_ctx = stripped[i + 1][:right_ov]
                if this_tail.shape[0] != next_left_ctx.shape[0]:
                    n = min(this_tail.shape[0], next_left_ctx.shape[0])
                    this_tail, next_left_ctx = this_tail[-n:], next_left_ctx[:n]
                blended, used_fallback = _crossfade(this_tail, next_left_ctx, blend_mode)
                if used_fallback:
                    flow_fallback_boundaries.append(f"{meta['index']}/{meta['index']+1}")
                result_pieces.append(blended)
            else:
                result_pieces.append(core)

        final = torch.cat(result_pieces, dim=0)

        if flow_fallback_boundaries:
            notes.append(
                "flow_align: opencv-python (cv2) не найден или не справился с "
                f"кадром на стыке(ах) {', '.join(flow_fallback_boundaries)} — "
                "для них тихо использован обычный smoothstep без выравнивания."
            )

        if final.shape[0] != total_frames:
            notes.append(f"WARNING: reconstructed length {final.shape[0]} != "
                          f"source total_frames {total_frames}")
        info = f"merged {num_chunks} chunks -> {final.shape[0]} frames " \
               f"(source was {total_frames}), blend_mode={blend_mode}\n" + \
               ("\n".join(notes) if notes else "no issues")

        return (final, info)


# --------------------------------------------------------------------------
# 5. Audio companions to the video Splitter/Merge — cuts and reassembles
#    the AUDIO track at EXACTLY the same timeline positions as the video
#    chunks (overlap included), so a lipsync-aware / audio-conditioned
#    second-pass sampler gets synchronized audio per chunk instead of
#    silence or a naively-chopped track.
# --------------------------------------------------------------------------
def _slice_waveform_padded(waveform, start, end, total_samples):
    """waveform: [B,C,S]. Returns waveform[...,start:end], padding by
    holding the edge sample if start<0 or end>total_samples."""
    start_c = max(0, start)
    end_c = min(total_samples, end)
    piece = waveform[..., start_c:end_c]
    missing_left = start_c - start
    missing_right = end - end_c
    if missing_left > 0:
        if piece.shape[-1] > 0:
            pad = piece[..., :1].repeat(1, 1, missing_left)
        else:
            pad = torch.zeros(waveform.shape[0], waveform.shape[1], missing_left,
                               device=waveform.device, dtype=waveform.dtype)
        piece = torch.cat([pad, piece], dim=-1)
    if missing_right > 0:
        if piece.shape[-1] > 0:
            pad = piece[..., -1:].repeat(1, 1, missing_right)
        else:
            pad = torch.zeros(waveform.shape[0], waveform.shape[1], missing_right,
                               device=waveform.device, dtype=waveform.dtype)
        piece = torch.cat([piece, pad], dim=-1)
    return piece, missing_left, missing_right


class MMH3_AudioChunkSplitter:
    """
    Cuts an AUDIO track at exactly the same timeline positions as
    MMH3 Chunk Splitter cuts the video (same overlap regions, same
    edge/pad handling philosophy), so each video chunk's second-pass
    sampler can be given audio that is actually in sync with it —
    important for anything lipsync/audio-conditioned.

    Computes its own sample-accurate plan (`audio_plan`) from the video
    `plan`, using cumulative rounding so the core segments always sum to
    exactly the source audio's sample count — same exactness guarantee
    as the video path.
    """

    DESCRIPTION = (
        "Режет звуковую дорожку ровно по тем же временным границам, что "
        "MMH3 Chunk Splitter режет видео (тот же нахлёст на стыках). Нужно, "
        "если второй проход семплером зависит от звука (например, липсинк). "
        "Выдаёт свой audio_plan — его нужно довести до MMH3 Audio Chunk Merge."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Полная звуковая дорожка исходного ролика MiniMax H3 (тот же по длительности, что и видео)."}),
                "plan": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Тот же plan из MMH3 Chunk Splitter (видео) — определяет, где проходят границы кусков и нахлёстов."}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio_chunk_1", "audio_chunk_2", "audio_chunk_3", "audio_chunk_4", "audio_chunk_5", "audio_plan", "info")
    OUTPUT_TOOLTIPS = (
        "Звук куска 1, синхронный с chunk_1 из видео-Splitter'а. Подать в тот же второй проход, что и видео-кусок.",
        "Звук куска 2. Есть, если num_chunks >= 2.",
        "Звук куска 3. Есть, если num_chunks >= 3.",
        "Звук куска 4. Есть, если num_chunks >= 4.",
        "Звук куска 5. Есть, только если num_chunks = 5.",
        "JSON-карта звуковой разбивки — обязательно довести до MMH3 Audio Chunk Merge.audio_plan.",
        "Текстовый отчёт по звуковой разбивке.",
    )
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Seamless Chunks"

    def run(self, audio, plan):
        payload = json.loads(plan)
        chunks_meta = payload["chunks"]
        total_frames = payload["total_frames"]

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        total_samples = waveform.shape[-1]
        rate = total_samples / float(total_frames)

        # cumulative, drift-free core boundaries in sample-space
        cum = 0
        sample_pos = [0]
        for meta in chunks_meta:
            cum += meta["core_len"]
            sample_pos.append(int(round(cum * rate)))

        notes = []
        outputs = []
        audio_chunks_out = []
        for i, meta in enumerate(chunks_meta):
            core_start_s = sample_pos[i]
            core_end_s = sample_pos[i + 1]
            core_len_s = core_end_s - core_start_s
            left_ov_s = int(round(meta["left_ov"] * rate)) if meta["left_ov"] > 0 else 0
            right_ov_s = int(round(meta["right_ov"] * rate)) if meta["right_ov"] > 0 else 0
            raw_start_s = core_start_s - left_ov_s
            raw_end_s = core_end_s + right_ov_s
            pad_s = int(round(meta["pad"] * rate)) if meta["pad"] > 0 else 0

            piece, missing_left, missing_right = _slice_waveform_padded(
                waveform, raw_start_s, raw_end_s, total_samples)
            if missing_left or missing_right:
                notes.append(f"chunk {meta['index']}: audio padded {missing_left}+{missing_right} "
                              f"sample(s) at edges (source audio shorter than expected)")

            raw_len_s = piece.shape[-1]
            if pad_s > 0:
                tail = piece[..., -1:].repeat(1, 1, pad_s)
                piece = torch.cat([piece, tail], dim=-1)
                notes.append(f"chunk {meta['index']}: +{pad_s} audio sample(s) (held) to match video padding")

            audio_chunks_out.append({"waveform": piece, "sample_rate": sample_rate})
            audio_chunks_out_meta = dict(
                index=meta["index"], core_len_samples=core_len_s,
                left_ov_samples=left_ov_s, right_ov_samples=right_ov_s,
                raw_len_samples=raw_len_s, pad_samples=pad_s,
                final_len_samples=raw_len_s + pad_s,
            )
            outputs.append(audio_chunks_out_meta)

        while len(audio_chunks_out) < 5:
            audio_chunks_out.append({"waveform": waveform[..., :1], "sample_rate": sample_rate})

        audio_plan = json.dumps({
            "total_samples": total_samples,
            "sample_rate": sample_rate,
            "chunks": outputs,
        })
        info = f"total_samples={total_samples} sample_rate={sample_rate}\n" + \
               ("\n".join(notes) if notes else "no padding needed")

        return (audio_chunks_out[0], audio_chunks_out[1], audio_chunks_out[2],
                audio_chunks_out[3], audio_chunks_out[4], audio_plan, info)


def _audio_crossfade(a, b, mode):
    """a, b: [B,C,N] waveform tensors, equal N along last dim."""
    n = a.shape[-1]
    if n == 0:
        return a
    device, dtype = a.device, a.dtype
    t = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).view(1, 1, n)
    if mode == "equal_power":
        alpha = torch.sqrt(t)
        beta = torch.sqrt(1 - t)
        return a * beta + b * alpha
    return a * (1 - t) + b * t  # linear


class MMH3_AudioChunkMerge:
    """
    Reassembles processed audio chunks back into one track using the
    exact sample-accurate boundaries recorded in `audio_plan` by
    MMH3 Audio Chunk Splitter — same crossfade-the-overlap philosophy as
    the video merge, so an audio edit made by your second pass (if any)
    doesn't click at the seams. If your second pass leaves audio
    untouched, you don't need this node at all — just remux the
    original full-length audio onto the final merged video.
    """

    DESCRIPTION = (
        "Собирает звуковые куски обратно в одну дорожку, число сэмплов на "
        "выходе точно равно исходному. Нужна только если второй проход "
        "семплером сам меняет звук (например, совместная аудио-видео "
        "генерация с липсинком). Если звук не меняется — просто "
        "замьюксируйте исходную полную дорожку поверх готового видео."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_plan": ("STRING", {
                    "forceInput": True,
                    "tooltip": "audio_plan из MMH3 Audio Chunk Splitter. Обязателен."}),
                "audio_chunk_1": ("AUDIO", {
                    "tooltip": "Обработанный звук куска 1. Обязателен всегда."}),
                "blend_mode": (["equal_power", "linear"], {
                    "default": "equal_power",
                    "tooltip": "Кроссфейд звука на стыке: equal_power — стандартный аудио-кроссфейд без провала громкости в середине (рекомендуется); linear — простое линейное смешивание."}),
            },
            "optional": {
                "audio_chunk_2": ("AUDIO", {"tooltip": "Звук куска 2. Нужен, если чанков >= 2."}),
                "audio_chunk_3": ("AUDIO", {"tooltip": "Звук куска 3. Нужен, если чанков >= 3."}),
                "audio_chunk_4": ("AUDIO", {"tooltip": "Звук куска 4. Нужен, если чанков >= 4."}),
                "audio_chunk_5": ("AUDIO", {"tooltip": "Звук куска 5. Нужен, только если чанков = 5."}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    OUTPUT_TOOLTIPS = (
        "Готовая склеенная звуковая дорожка, число сэмплов = исходному. Замьюксировать с images из MMH3 Chunk Merge.",
        "Текстовый отчёт по звуковой склейке.",
    )
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Seamless Chunks"

    def run(self, audio_plan, audio_chunk_1, blend_mode, audio_chunk_2=None,
            audio_chunk_3=None, audio_chunk_4=None, audio_chunk_5=None):
        payload = json.loads(audio_plan)
        chunks_meta = payload["chunks"]
        total_samples = payload["total_samples"]
        sample_rate = payload["sample_rate"]
        num_chunks = len(chunks_meta)

        raw_inputs = [audio_chunk_1, audio_chunk_2, audio_chunk_3, audio_chunk_4, audio_chunk_5][:num_chunks]
        for i, a in enumerate(raw_inputs):
            if a is None:
                raise ValueError(f"audio_chunk_{i+1} is required (audio_plan expects {num_chunks} chunks) but was not connected")

        waveforms = [a["waveform"] for a in raw_inputs]

        stripped = []
        notes = []
        for meta, wf in zip(chunks_meta, waveforms):
            expected = meta["final_len_samples"]
            if wf.shape[-1] != expected:
                notes.append(f"chunk {meta['index']}: expected {expected} audio samples, got {wf.shape[-1]}")
            raw_len = meta["raw_len_samples"]
            piece = wf[..., :raw_len]
            if piece.shape[-1] < raw_len:
                pad = piece[..., -1:].repeat(1, 1, raw_len - piece.shape[-1])
                piece = torch.cat([piece, pad], dim=-1)
            stripped.append(piece)

        result_pieces = []
        for i, meta in enumerate(chunks_meta):
            piece = stripped[i]
            left_ov = meta["left_ov_samples"]
            right_ov = meta["right_ov_samples"]
            core_len = meta["core_len_samples"]
            core = piece[..., left_ov: left_ov + core_len]

            if right_ov > 0:
                keep_len = core_len - right_ov
                result_pieces.append(core[..., :keep_len])
                this_tail = core[..., keep_len:]
                next_left_ctx = stripped[i + 1][..., :right_ov]
                if this_tail.shape[-1] != next_left_ctx.shape[-1]:
                    n = min(this_tail.shape[-1], next_left_ctx.shape[-1])
                    this_tail, next_left_ctx = this_tail[..., -n:], next_left_ctx[..., :n]
                result_pieces.append(_audio_crossfade(this_tail, next_left_ctx, blend_mode))
            else:
                result_pieces.append(core)

        final = torch.cat(result_pieces, dim=-1)
        if final.shape[-1] != total_samples:
            notes.append(f"WARNING: reconstructed length {final.shape[-1]} samples != "
                          f"source total_samples {total_samples}")

        info = f"merged {num_chunks} audio chunks -> {final.shape[-1]} samples " \
               f"(source was {total_samples}), blend_mode={blend_mode}\n" + \
               ("\n".join(notes) if notes else "no issues")

        return ({"waveform": final, "sample_rate": sample_rate}, info)


# --------------------------------------------------------------------------
# 6. Latent-space variant of the Splitter/Merge pair.
#
# MiniMax H3 stores an AV NestedTensor in latent["samples"]:
#   video [B, 24, T, H/16, W/16]   time = dim 2
#   audio [B, 32, 2, Ta]           time = dim 3
# Video tokens follow FRAME_PER_TOKEN = (1, 4, 4, 4, 4): token 0 of every
# 5-token block is a 1-frame keyframe, T must be 5k+2, pixel frames 17k+5.
#
# Slicing NestedTensor with [:, :, start:end] applies the SAME index to
# audio dim 2 (stereo, size 2) — chunk 1 keeps audio by accident, chunk 2
# does not — AND starts chunk 2 off the keyframe grid, so VAEDecode of
# that slice pulses at 17 frames. Both are fixed here: unbind, slice each
# stream on its real time axis, snap starts to %5==0, pad each chunk to
# 5k+2, rebind NestedTensor.
# --------------------------------------------------------------------------
def _slice_latent_chunk(video, audio, c, notes):
    """Cut one planned window out of (video, audio), pad edges + tail.

    Returns (piece_video, piece_audio).
    """
    T = int(video.shape[2])
    Ta = int(audio.shape[-1]) if audio is not None else 0
    start, end = c["raw_start"], c["raw_end"]
    v, a, t0c, t1c, a0, a1 = slice_av_window(video, audio, start, end, Ta or None)

    missing_left = t0c - start
    missing_right = end - t1c
    if missing_left > 0:
        v = pad_video_head(v, missing_left)
        if a is not None:
            f_left = frames_for_tokens(t0c) - frames_for_tokens(max(0, start))
            hold_a = max(0, int(round(f_left * 5.0 / 3.0)))
            a = pad_audio_head(a, hold_a)
        notes.append(
            f"chunk {c['index']}: padded {missing_left} latent token(s) at left edge"
        )
    if missing_right > 0:
        v = pad_video_tail(v, missing_right)
        if a is not None:
            hold_a = max(0, int(round((end - t1c) * (Ta / max(T, 1)))))
            a = pad_audio_tail(a, hold_a)
        notes.append(
            f"chunk {c['index']}: padded {missing_right} latent token(s) at right edge"
        )

    if c["pad"] > 0:
        v = pad_video_tail(v, c["pad"])
        if a is not None:
            hold_a = max(0, int(round(c["pad"] * (Ta / max(T, 1)))))
            a = pad_audio_tail(a, hold_a)
        notes.append(
            f"chunk {c['index']}: +{c['pad']} duplicated latent token(s) "
            f"to satisfy 5n+2 "
            f"({c['raw_len']} -> {c['final_len']})"
        )
    return v, a


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
            v, a = _slice_latent_chunk(video, audio, c, notes)
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
            # mixed — still OK, we pack based on what we actually unbound
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
                # Audio pad was proportional; after processing, keep whatever
                # length we have and trim using the same left/right ov ratio
                # as video in the stitch loop (handled there via audio_token_range
                # on the *source* window, not the processed length).
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
                # Map the same video-token windows onto audio via the plan's
                # raw_start/raw_end, which is where the splitter cut audio.
                # After independent processing the audio length may have
                # drifted — fall back to proportional slices of the piece.
                Ta = int(piece_a.shape[-1])
                raw_len_v = meta["raw_len"]
                def _a_span(tok0, tok1):
                    # tok0/tok1 are offsets inside this piece (0..raw_len)
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


# --------------------------------------------------------------------------
# 7. Small utility: grab the last N frames of a clip, for reuse as
#    reference images / first-frame conditioning in the next MiniMax H3
#    generation (chaining scenes, or re-feeding the same reference set
#    into each chunk's second-pass sampler).
# --------------------------------------------------------------------------
class MMH3_LastFrames:
    """Extracts the last N frames of an IMAGE batch, e.g. to reuse as a
    reference/conditioning image for the next generation or for every
    chunk's second-pass sampler."""

    DESCRIPTION = (
        "Берёт последние N кадров ролика — удобно, чтобы прокинуть тот же "
        "референс/conditioning в следующую генерацию MiniMax H3 или во все "
        "четыре ветки второго прохода семплером, сохраняя единый стиль."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Ролик (IMAGE-батч), из которого нужно взять последние кадры."}),
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


NODE_CLASS_MAPPINGS = {
    "MMH3_FrameCalculator": MMH3_FrameCalculator,
    "MMH3_ChunkSplitter": MMH3_ChunkSplitter,
    "MMH3_ChunkMerge": MMH3_ChunkMerge,
    "MMH3_LatentChunkSplitter": MMH3_LatentChunkSplitter,
    "MMH3_LatentChunkMerge": MMH3_LatentChunkMerge,
    "MMH3_LatentInfo": MMH3_LatentInfo,
    "MMH3_AudioChunkSplitter": MMH3_AudioChunkSplitter,
    "MMH3_AudioChunkMerge": MMH3_AudioChunkMerge,
    "MMH3_LastFrames": MMH3_LastFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MMH3_FrameCalculator": "🎬 MMH3 Frame Calculator",
    "MMH3_ChunkSplitter": "✂️ MMH3 Chunk Splitter (Seamless)",
    "MMH3_ChunkMerge": "🧵 MMH3 Chunk Merge (Seamless)",
    "MMH3_LatentChunkSplitter": "🧬 MMH3 Latent Chunk Splitter",
    "MMH3_LatentChunkMerge": "🧬 MMH3 Latent Chunk Merge",
    "MMH3_LatentInfo": "🔎 MMH3 Latent Info",
    "MMH3_AudioChunkSplitter": "🔊 MMH3 Audio Chunk Splitter",
    "MMH3_AudioChunkMerge": "🎚️ MMH3 Audio Chunk Merge",
    "MMH3_LastFrames": "⏮️ MMH3 Last Frames",
}
