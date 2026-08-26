"""
Frame-count planning for MiniMax H3 seamless chunked upscale/resample.

MiniMax H3 does not render an arbitrary number of frames: it quantizes the
requested duration up to the next value of the form

    frames = block * k + remainder      (block=17, remainder=5, fps=24)

e.g. a 10s request (240 frames @24fps) is rendered as 243 frames (17*14+5),
and a 15s request (360 frames) is rendered as 362 frames (17*21+5).

The VisualVAE is temporally causal (f16t4d24). Video latent tokens follow
the repeating coverage grid

    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

so token 0 of every 5-token block is a 1-frame KEYFRAME and the other four
tokens cover 4 pixel frames each. Consequently:

    pixel frames  = 17k + 5
    latent tokens =  5k + 2

A chunk that does NOT start on a keyframe (token index % 5 == 0) is decoded
on a phase-shifted timeline — the VAE treats its first token as a 1-frame
keyframe when it is actually a 4-frame residual. That is the 17-frame pulse
/ flicker after the first chunk.

This module:
  1. Reproduces the 17k+5 formula so you can know the *real* frame count.
  2. Plans a split of N frames into `num_chunks` pieces that share `overlap`
     frames with their neighbours, cores summing to exactly N.
  3. Optionally pads each chunk up to a sampler formula (4n+1, 5n+2, …).
  4. Plans H3-grid-aligned latent splits (starts on keyframes, overlap a
     multiple of 5, each chunk padded to 5k+2).
"""

import math


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
H3_TOKEN_BLOCK = 5
H3_TOKEN_REMAINDER = 2
H3_PIXEL_BLOCK = 17
H3_PIXEL_REMAINDER = 5
# Audio latent rate is 40 Hz vs 24 fps video → 40/24 = 5/3 audio tokens per frame.
FRAME_RESCALE = 5.0 / 3.0


def quantize_up(n: int, multiple: int, remainder: int) -> int:
    """Smallest value >= n of the form multiple*k + remainder (k>=0)."""
    if multiple <= 0:
        return max(n, remainder)
    if n <= remainder:
        return remainder
    k = math.ceil((n - remainder) / multiple)
    return multiple * k + remainder


def quantize_down(n: int, multiple: int, remainder: int) -> int:
    """Largest value <= n of the form multiple*k + remainder (k>=0), or remainder."""
    if multiple <= 0:
        return min(n, remainder) if n < remainder else remainder
    if n < remainder:
        return remainder if remainder <= n else n
    k = (n - remainder) // multiple
    return multiple * k + remainder


def minimax_h3_frame_count(seconds: float, fps: int = 24, block: int = 17, remainder: int = 5) -> int:
    """Real rendered frame count for a requested duration, per MiniMax H3's
    17k+5 quantization. Verified against public examples: 10s -> 243,
    15s -> 362."""
    n = round(seconds * fps)
    return quantize_up(n, block, remainder)


def frames_for_tokens(n: int) -> int:
    """Pixel frames covered by the first `n` video latent tokens."""
    if n <= 0:
        return 0
    return sum(FRAME_PER_TOKEN[i % 5] for i in range(n))


def tokens_for_frames(f: int) -> int:
    """Smallest token count whose cumulative frames reach at least `f`."""
    n, acc = 0, 0
    f = max(0, int(f))
    while acc < f:
        acc += FRAME_PER_TOKEN[n % 5]
        n += 1
    return n


def is_h3_token_grid(t: int) -> bool:
    """True iff t == 5k+2 for some k>=0."""
    return t >= H3_TOKEN_REMAINDER and (t - H3_TOKEN_REMAINDER) % H3_TOKEN_BLOCK == 0


def tokens_to_h3_frames(t: int) -> int:
    """Pixel frames implied by t latent tokens on the official grid.
    Only exact when is_h3_token_grid(t); otherwise returns frames_for_tokens(t)."""
    return frames_for_tokens(t)


def audio_token_range(token_start: int, token_end: int, total_audio: int | None = None):
    """Audio latent token range [a0, a1) covering video tokens [token_start, token_end).

    Uses pixel-frame coverage (not a naive T-proportional map) because H3 tokens
    have uneven duration: 1 frame, then 4, 4, 4, 4.
    """
    f0 = frames_for_tokens(max(0, token_start))
    f1 = frames_for_tokens(max(0, token_end))
    a0 = int(round(f0 * FRAME_RESCALE))
    a1 = int(round(f1 * FRAME_RESCALE))
    if total_audio is not None:
        a0 = max(0, min(total_audio, a0))
        a1 = max(a0, min(total_audio, a1))
    return a0, a1


def plan_chunks(total_frames: int, num_chunks: int, overlap: int,
                 pad_multiple: int = 4, pad_remainder: int = 1):
    """
    Split `total_frames` into `num_chunks` overlapping pieces.

    - The non-overlapping "core" pieces always sum to exactly total_frames.
    - Each piece additionally carries `overlap` frames of context borrowed
      from its neighbour on each internal boundary (first/last piece only
      have one side).
    - Each piece is then padded (by duplicating its own last frame) up to
      the nearest `pad_multiple * k + pad_remainder` length, so it's a
      valid input size for a second sampler pass with a fixed temporal
      compression factor. Padding is recorded so it can be stripped later.

    Returns a list of dicts, one per chunk, in order.
    """
    if num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    if total_frames < num_chunks:
        raise ValueError("total_frames smaller than num_chunks")

    base_core = total_frames // num_chunks
    rem = total_frames % num_chunks
    core_lens = [base_core + (1 if i < rem else 0) for i in range(num_chunks)]

    if overlap > 0 and min(core_lens) <= overlap:
        raise ValueError(
            f"overlap ({overlap}) too large for core length "
            f"({min(core_lens)}); reduce overlap or num_chunks"
        )

    chunks = []
    cursor = 0
    for i, c in enumerate(core_lens):
        left_ov = overlap if i > 0 else 0
        right_ov = overlap if i < num_chunks - 1 else 0
        raw_start = cursor - left_ov
        raw_end = cursor + c + right_ov
        raw_len = raw_end - raw_start
        final_len = quantize_up(raw_len, pad_multiple, pad_remainder)
        pad = final_len - raw_len
        chunks.append(dict(
            index=i,
            core_start=cursor, core_len=c,
            left_ov=left_ov, right_ov=right_ov,
            raw_start=raw_start, raw_end=raw_end, raw_len=raw_len,
            pad=pad, final_len=final_len,
            start_phase=raw_start % H3_TOKEN_BLOCK if raw_start >= 0 else raw_start,
        ))
        cursor += c

    assert cursor == total_frames
    return chunks


def plan_chunks_h3(total_tokens: int, num_chunks: int, overlap: int,
                    pad_multiple: int = 5, pad_remainder: int = 2):
    """
    Split `total_tokens` latent tokens into `num_chunks` pieces that each
    start on an H3 keyframe (token index % 5 == 0).

    Overlap is snapped DOWN to a multiple of 5 (one 17-frame block). Cores
    of every chunk except possibly the last are multiples of 5; the last
    core absorbs the +2 remainder of the 5k+2 grid so the cores still sum
    to exactly total_tokens. Each chunk is then padded up to 5k+2 so it is
    a valid standalone clip for the causal VAE / DiT.

    This is what stops the 17-frame pulse: chunk 2+ is no longer decoded
    on a phase-shifted (1,4,4,4,4) grouping.
    """
    if num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    if total_tokens < num_chunks:
        raise ValueError("total_tokens smaller than num_chunks")

    # Snap overlap to a multiple of 5. Values in (0, 5) — including the old
    # default of 2 — round UP to 5, otherwise a leftover workflow would
    # silently get zero overlap and a hard cut at the keyframe.
    if overlap <= 0:
        ov = 0
    else:
        snapped = int(round(overlap / float(H3_TOKEN_BLOCK))) * H3_TOKEN_BLOCK
        ov = max(H3_TOKEN_BLOCK, snapped)

    body = (total_tokens // H3_TOKEN_BLOCK) * H3_TOKEN_BLOCK
    rem = total_tokens - body
    groups = body // H3_TOKEN_BLOCK
    if groups < num_chunks:
        raise ValueError(
            f"total_tokens={total_tokens} too short to place {num_chunks} "
            f"keyframe-aligned chunks (need at least {num_chunks * H3_TOKEN_BLOCK} "
            f"tokens of body). Reduce num_chunks."
        )

    base_g = groups // num_chunks
    extra_g = groups % num_chunks
    core_lens = [
        H3_TOKEN_BLOCK * (base_g + (1 if i < extra_g else 0))
        for i in range(num_chunks)
    ]
    core_lens[-1] += rem

    if ov > 0:
        for i, c in enumerate(core_lens):
            right = ov if i < num_chunks - 1 else 0
            left = ov if i > 0 else 0
            # keep_len = core_len - right_ov must stay > 0 so the chunk
            # actually owns some unique timeline.
            if right and c <= right:
                raise ValueError(
                    f"overlap ({ov} tokens, snapped from {overlap}) too large "
                    f"for core length {c} of chunk {i}; reduce overlap or num_chunks"
                )
            if left and c <= 0:
                raise ValueError(
                    f"chunk {i} has empty core after H3 grid snap"
                )

    chunks = []
    cursor = 0
    for i, c in enumerate(core_lens):
        left_ov = ov if i > 0 else 0
        right_ov = ov if i < num_chunks - 1 else 0
        raw_start = cursor - left_ov
        raw_end = cursor + c + right_ov
        raw_len = raw_end - raw_start
        final_len = quantize_up(raw_len, pad_multiple, pad_remainder)
        pad = final_len - raw_len
        phase = raw_start % H3_TOKEN_BLOCK if raw_start >= 0 else raw_start
        chunks.append(dict(
            index=i,
            core_start=cursor, core_len=c,
            left_ov=left_ov, right_ov=right_ov,
            raw_start=raw_start, raw_end=raw_end, raw_len=raw_len,
            pad=pad, final_len=final_len,
            start_phase=phase,
            on_keyframe=phase == 0,
            final_on_grid=is_h3_token_grid(final_len),
        ))
        cursor += c

    assert cursor == total_tokens
    return chunks
