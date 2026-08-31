"""Split→merge identity + NestedTensor-slicing bug reproduction.

Requires torch. Skipped automatically if torch is not installed.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))  # so ComfyUI-MiniMaxSeamlessChunks is importable as a package

try:
    import torch
except ImportError:
    torch = None


def _load():
    """Load the pack as a namespace without requiring comfy."""
    import types
    pkg = types.ModuleType("ComfyUI_MiniMaxSeamlessChunks")
    pkg.__path__ = [str(ROOT)]
    sys.modules["ComfyUI_MiniMaxSeamlessChunks"] = pkg
    sys.modules["ComfyUI-MiniMaxSeamlessChunks"] = pkg
    # The package uses relative imports (from .plan). Expose as that name.
    # Easier: exec modules with package context.
    import importlib.util

    def load(name, file):
        spec = importlib.util.spec_from_file_location(
            f"ComfyUI_MiniMaxSeamlessChunks.{name}", ROOT / file,
            submodule_search_locations=[str(ROOT)],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    pkg.plan = load("plan", "plan.py")
    pkg.av_latent = load("av_latent", "av_latent.py")
    return pkg


@unittest.skipIf(torch is None, "torch not installed")
class TestNestedSliceBug(unittest.TestCase):
    def setUp(self):
        self.pkg = _load()

    def _make_av(self, T=72, Ta=405):
        video = torch.arange(T, dtype=torch.float32).view(1, 1, T, 1, 1).repeat(1, 24, 1, 2, 2)
        # stamp each token with its index so we can detect wrong slices
        for t in range(T):
            video[:, :, t] = t
        audio = torch.arange(Ta, dtype=torch.float32).view(1, 1, 1, Ta).repeat(1, 32, 2, 1)
        for a in range(Ta):
            audio[..., a] = a
        return video, audio

    def test_naive_nested_getitem_breaks_audio_and_phase(self):
        """Reproduce the original bug: samples[:,:,s:e] on a NestedTensor."""
        video, audio = self._make_av()
        Nested = self.pkg.av_latent._FallbackNested
        samples = Nested((video, audio))

        # What the old splitter did: samples[:, :, start:end]
        # FallbackNested doesn't implement __getitem__, which is itself the
        # point — we must unbind. Simulate the typical NestedTensor.__getitem__
        # that maps the same index onto every member:
        def naive_slice(nested, s, e):
            return Nested([t[:, :, s:e] for t in nested.unbind()])

        chunk2 = naive_slice(samples, 34, 72)  # old planner, overlap=2, 2 chunks
        v2, a2 = chunk2.unbind()
        # video starts at token 34 → phase 4, off keyframe
        self.assertEqual(int(v2[0, 0, 0, 0, 0].item()), 34)
        self.assertEqual(34 % 5, 4)
        # audio dim 2 is stereo (size 2); [:,:,34:72] is EMPTY
        self.assertEqual(a2.shape[2], 0)

    def test_unbind_slice_keeps_audio_time_and_keyframe(self):
        video, audio = self._make_av()
        av = self.pkg.av_latent
        plan = self.pkg.plan
        chunks = plan.plan_chunks_h3(72, 2, 5)
        # chunk 1 start must be on keyframe
        self.assertEqual(chunks[1]["raw_start"] % 5, 0)
        v, a, t0, t1, a0, a1 = av.slice_av_window(
            video, audio, chunks[1]["raw_start"], chunks[1]["raw_end"], audio.shape[-1]
        )
        self.assertEqual(int(v[0, 0, 0, 0, 0].item()) % 5, 0)
        self.assertGreater(a.shape[-1], 0)
        self.assertEqual(a.shape[2], 2)  # stereo intact


@unittest.skipIf(torch is None, "torch not installed")
class TestSplitMergeIdentity(unittest.TestCase):
    def setUp(self):
        self.pkg = _load()
        # Import nodes with a fake `torch` already present; nodes also import
        # json/numpy. numpy may be missing too.
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")

    def test_h3_split_merge_identity_plain_video(self):
        from ComfyUI_MiniMaxSeamlessChunks import av_latent as av
        from ComfyUI_MiniMaxSeamlessChunks import plan as plan
        # Use the node helpers directly rather than instantiating Comfy node
        # classes (they need ComfyUI widget metadata).
        T = 72
        video = torch.zeros(1, 24, T, 4, 4)
        for t in range(T):
            video[:, :, t] = t
        audio = torch.zeros(1, 32, 2, 405)
        for i in range(405):
            audio[..., i] = i

        chunks = plan.plan_chunks_h3(T, 2, 5)
        notes = []
        pieces_v = []
        pieces_a = []
        # inline the same stitch as the merge (causal = identity on unprocessed)
        stripped_v = []
        stripped_a = []
        for c in chunks:
            v, a, t0, t1, a0, a1 = av.slice_av_window(
                video, audio, c["raw_start"], c["raw_end"], 405
            )
            # no edge pad for this on-grid case
            if c["pad"] > 0:
                v = av.pad_video_tail(v, c["pad"])
                a = av.pad_audio_tail(a, max(0, int(round(c["pad"] * (405 / T)))))
            # strip pad
            v = v[:, :, : c["raw_len"]]
            stripped_v.append(v)
            stripped_a.append(a)

        result_v = []
        for i, meta in enumerate(chunks):
            piece = stripped_v[i]
            left_ov, right_ov, core_len = meta["left_ov"], meta["right_ov"], meta["core_len"]
            core = piece[:, :, left_ov: left_ov + core_len]
            if right_ov > 0:
                keep = core_len - right_ov
                result_v.append(core[:, :, :keep])
                result_v.append(core[:, :, keep:])  # causal: keep earlier tail
            else:
                result_v.append(core)
        merged = av.cat_video(result_v)
        self.assertEqual(int(merged.shape[2]), T)
        self.assertTrue(torch.equal(merged, video))


if __name__ == "__main__":
    unittest.main()
