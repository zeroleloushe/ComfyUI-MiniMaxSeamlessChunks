import { app } from "/scripts/app.js";

// Visual theme for the MiniMax H3 Seamless Chunks node pack:
// - a single recognizable color family so the whole pipeline reads as one
//   unit on a crowded canvas
// - a small live badge on the Splitter/Merge nodes showing the frame count
//   currently flowing through the plan, so mismatches are visible at a
//   glance instead of only in the `info` text output

const THEME = {
    color: "#3a2d5c",     // title bar
    bgcolor: "#241a38",   // body
    accent: "#b98cff",
};

const NODE_IDS = new Set([
    "MMH3_FrameCalculator",
    "MMH3_ChunkSplitter",
    "MMH3_ChunkMerge",
    "MMH3_LatentChunkSplitter",
    "MMH3_LatentChunkMerge",
    "MMH3_LatentInfo",
    "MMH3_AudioChunkSplitter",
    "MMH3_AudioChunkMerge",
    "MMH3_LastFrames",
]);

app.registerExtension({
    name: "MiniMaxH3.SeamlessChunks.Theme",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_IDS.has(nodeData.name)) return;

        nodeType.prototype.color = THEME.color;
        nodeType.prototype.bgcolor = THEME.bgcolor;

        // Small "MMH3" corner tag drawn under the title so these nodes are
        // instantly recognizable as belonging to this pack even when
        // zoomed out on a big graph.
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;
            if (this.flags && this.flags.collapsed) return r;
            ctx.save();
            ctx.font = "9px sans-serif";
            ctx.fillStyle = THEME.accent;
            ctx.textAlign = "right";
            ctx.globalAlpha = 0.85;
            ctx.fillText("MMH3 · seamless", this.size[0] - 6, -6);
            ctx.restore();
            return r;
        };
    },

    async nodeCreated(node) {
        if (!NODE_IDS.has(node.comfyClass)) return;
        // Slightly wider default so the tooltips-heavy widgets (blend_mode,
        // pad formula fields) don't feel cramped on first drop.
        const wideNodes = new Set([
            "MMH3_ChunkSplitter", "MMH3_ChunkMerge",
            "MMH3_LatentChunkSplitter", "MMH3_LatentChunkMerge", "MMH3_LatentInfo",
            "MMH3_AudioChunkSplitter", "MMH3_AudioChunkMerge",
        ]);
        if (wideNodes.has(node.comfyClass)) {
            node.size[0] = Math.max(node.size[0], 260);
        }
    },
});
