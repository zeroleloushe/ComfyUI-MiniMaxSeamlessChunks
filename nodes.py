from .latent_split import MMH3_LatentChunkSplitter
from .latent_merge import MMH3_LastFrames, MMH3_LatentChunkMerge, MMH3_LatentInfo

try:
    from .pixel_nodes import (
        MMH3_AudioChunkMerge,
        MMH3_AudioChunkSplitter,
        MMH3_ChunkMerge,
        MMH3_ChunkSplitter,
        MMH3_FrameCalculator,
    )
except ImportError:
    MMH3_AudioChunkMerge = None
    MMH3_AudioChunkSplitter = None
    MMH3_ChunkMerge = None
    MMH3_ChunkSplitter = None
    MMH3_FrameCalculator = None

NODE_CLASS_MAPPINGS = {
    "MMH3_LatentChunkSplitter": MMH3_LatentChunkSplitter,
    "MMH3_LatentChunkMerge": MMH3_LatentChunkMerge,
    "MMH3_LatentInfo": MMH3_LatentInfo,
    "MMH3_LastFrames": MMH3_LastFrames,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MMH3_LatentChunkSplitter": "🧬 MMH3 Latent Chunk Splitter",
    "MMH3_LatentChunkMerge": "🧬 MMH3 Latent Chunk Merge",
    "MMH3_LatentInfo": "🔎 MMH3 Latent Info",
    "MMH3_LastFrames": "⏮️ MMH3 Last Frames",
}

if MMH3_FrameCalculator is not None:
    NODE_CLASS_MAPPINGS.update({
        "MMH3_FrameCalculator": MMH3_FrameCalculator,
        "MMH3_ChunkSplitter": MMH3_ChunkSplitter,
        "MMH3_ChunkMerge": MMH3_ChunkMerge,
        "MMH3_AudioChunkSplitter": MMH3_AudioChunkSplitter,
        "MMH3_AudioChunkMerge": MMH3_AudioChunkMerge,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "MMH3_FrameCalculator": "🎬 MMH3 Frame Calculator",
        "MMH3_ChunkSplitter": "✂️ MMH3 Chunk Splitter (Seamless)",
        "MMH3_ChunkMerge": "🧵 MMH3 Chunk Merge (Seamless)",
        "MMH3_AudioChunkSplitter": "🔊 MMH3 Audio Chunk Splitter",
        "MMH3_AudioChunkMerge": "🎛️ MMH3 Audio Chunk Merge",
    })
