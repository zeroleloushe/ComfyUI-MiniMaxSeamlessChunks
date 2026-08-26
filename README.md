# ComfyUI-MiniMaxSeamlessChunks

Ноды ComfyUI для MiniMax H3: бесшовная нарезка/склейка латента по сетке keyframe `5k+2` (без 17-кадрового мерцания).

Репозиторий: https://github.com/zeroleloushe/ComfyUI-MiniMaxSeamlessChunks

## Установка

```
cd ComfyUI/custom_nodes
git clone https://github.com/zeroleloushe/ComfyUI-MiniMaxSeamlessChunks
```

Или ComfyUI Manager → Install via Git URL. Рестарт ComfyUI, Ctrl+F5.

## Ноды

| Нода | Зачем |
|---|---|
| 🧬 MMH3 Latent Chunk Splitter / Merge | Режет NestedTensor или video-only 5D по keyframe. Merge `causal`. |
| 🔎 MMH3 Latent Info | nested?, T, `5k+2`, фазы |
| ⏮️ MMH3 Last Frames | Хвост ролика в референс |

Дефолты: `align_h3_grid=ON`, `overlap=5`, Merge **causal**. Куски не декодировать отдельно — Merge → один VAEDecode.

Video-only после Split AV — нормальный вход, режет dim 2.
