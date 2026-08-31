"""Grid / planner tests — no torch, no ComfyUI required."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Import plan.py as a loose module (it has no package deps).
import importlib.util
spec = importlib.util.spec_from_file_location(
    "plan", Path(__file__).resolve().parents[1] / "plan.py"
)
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


class TestH3Grid(unittest.TestCase):
    def test_10s_tokens_and_frames(self):
        # 10s → 243 pixel frames → 72 latent tokens
        self.assertEqual(plan.minimax_h3_frame_count(10), 243)
        self.assertEqual(plan.frames_for_tokens(72), 243)
        self.assertEqual(plan.tokens_for_frames(243), 72)
        self.assertTrue(plan.is_h3_token_grid(72))

    def test_15s_tokens_and_frames(self):
        self.assertEqual(plan.minimax_h3_frame_count(15), 362)
        self.assertEqual(plan.frames_for_tokens(107), 362)
        self.assertEqual(plan.tokens_for_frames(362), 107)
        self.assertTrue(plan.is_h3_token_grid(107))

    def test_token_coverage_pattern(self):
        # (1,4,4,4,4) repeating
        self.assertEqual(plan.frames_for_tokens(1), 1)
        self.assertEqual(plan.frames_for_tokens(2), 5)
        self.assertEqual(plan.frames_for_tokens(5), 17)
        self.assertEqual(plan.frames_for_tokens(6), 18)
        self.assertEqual(plan.frames_for_tokens(10), 34)

    def test_5k2_membership(self):
        for k in range(0, 25):
            self.assertTrue(plan.is_h3_token_grid(5 * k + 2))
        for t in (0, 1, 3, 4, 5, 6, 8, 40, 41, 73):
            self.assertFalse(plan.is_h3_token_grid(t))


class TestPlanChunksH3(unittest.TestCase):
    def _check(self, total, n, overlap):
        chunks = plan.plan_chunks_h3(total, n, overlap)
        self.assertEqual(sum(c["core_len"] for c in chunks), total)
        self.assertEqual(len(chunks), n)
        self.assertEqual(chunks[-1]["core_start"] + chunks[-1]["core_len"], total)
        ov_snapped = (max(0, overlap) // 5) * 5
        for i, c in enumerate(chunks):
            if c["raw_start"] >= 0:
                self.assertEqual(
                    c["raw_start"] % 5, 0,
                    f"chunk {i} raw_start={c['raw_start']} not on keyframe",
                )
                self.assertTrue(c["on_keyframe"])
            self.assertEqual(c["start_phase"], 0)
            self.assertTrue(
                plan.is_h3_token_grid(c["final_len"]),
                f"chunk {i} final_len={c['final_len']} not 5k+2",
            )
            if i > 0:
                self.assertEqual(c["left_ov"], ov_snapped)
            if i < n - 1:
                self.assertEqual(c["right_ov"], ov_snapped)
                self.assertGreater(c["core_len"], ov_snapped)
        return chunks

    def test_10s_two_chunks_overlap5(self):
        chunks = self._check(72, 2, 5)
        # starts 0 and 35 (35%5==0), last core absorbs +2
        self.assertEqual(chunks[0]["core_start"], 0)
        self.assertEqual(chunks[1]["core_start"] % 5, 0)

    def test_10s_four_chunks(self):
        self._check(72, 4, 5)

    def test_15s_two_and_four(self):
        self._check(107, 2, 5)
        self._check(107, 4, 5)
        self._check(107, 5, 5)

    def test_overlap_snaps_down_to_multiple_of_5(self):
        chunks = plan.plan_chunks_h3(72, 2, 7)  # 7 → 5
        self.assertEqual(chunks[0]["right_ov"], 5)
        self.assertEqual(chunks[1]["left_ov"], 5)

    def test_old_default_overlap_2_rounds_up_to_5(self):
        chunks = plan.plan_chunks_h3(72, 2, 2)
        self.assertEqual(chunks[0]["right_ov"], 5)
        self.assertEqual(chunks[1]["raw_start"] % 5, 0)

    def test_old_planner_is_off_grid(self):
        """The bug: overlap=2, equal split, chunk 2 does not start on keyframe."""
        chunks = plan.plan_chunks(72, 2, 2, pad_multiple=1, pad_remainder=0)
        self.assertEqual(chunks[0]["raw_start"], 0)
        # core = 36, overlap 2 → chunk 1 raw_start = 36-2 = 34, 34%5 = 4
        self.assertEqual(chunks[1]["raw_start"] % 5, 4)
        self.assertFalse(plan.is_h3_token_grid(chunks[1]["raw_len"]))


class TestAudioRange(unittest.TestCase):
    def test_audio_follows_pixel_frames_not_token_index(self):
        # first token covers 1 frame, not 4 — proportional-to-T would be wrong
        a0, a1 = plan.audio_token_range(0, 1)
        self.assertEqual(a0, 0)
        self.assertEqual(a1, round(1 * 5 / 3))
        a0, a1 = plan.audio_token_range(0, 5)
        self.assertEqual(a1, round(17 * 5 / 3))


if __name__ == "__main__":
    unittest.main()
