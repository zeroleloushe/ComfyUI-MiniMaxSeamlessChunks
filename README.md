# ComfyUI-MiniMaxSeamlessChunks

**v1.1.0** · category `MiniMax H3 / Seamless Chunks` · [github.com/zeroleloushe/ComfyUI-MiniMaxSeamlessChunks](https://github.com/zeroleloushe/ComfyUI-MiniMaxSeamlessChunks)

ComfyUI custom nodes that take a finished MiniMax H3 clip (pixels **or** AV latent) plus its audio, split it into 2–5 overlapping chunks for parallel upscale / second-pass sampling, then stitch everything back so the output length **exactly** matches the source — pixel frames, latent tokens, or audio samples.

> **What's new in 1.1.0.** Pixel + audio splitters/merges and Frame Calculator now ship in the same pack as the latent path. `🧬 MMH3 Latent Chunk Splitter` is registered by default (no optional import). All nine nodes load from a single `nodes.py`. Tests for the `5k+2` grid planner are included.

English overview below, then the full Russian docs (graphs, I/O tables, flicker post-mortem, FAQ).

---

## What this pack actually does

MiniMax H3 does **not** emit an arbitrary number of frames. Duration is quantized:

| request | pixel frames (`17k+5` @24fps) | latent tokens (`5k+2`) |
|---|---|---|
| 10 s | 243 | 72 |
| 15 s | 362 | 107 |

The VisualVAE is **causal** (`f16t4d24`). Video tokens follow `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`: token 0 of every 5-token block is a 1-frame **keyframe**, the other four cover 4 pixel frames each. A chunk that does **not** start on a keyframe (`token % 5 == 0`) is decoded on a phase-shifted timeline — the VAE treats a 4-frame residual as a 1-frame keyframe. That is the **17-frame pulse / flicker** after chunk 1.

This pack:

1. Reproduces the `17k+5` formula (`🎬 Frame Calculator`) so you know the real frame count before generation.
2. Splits **IMAGE** batches with pixel-frame overlap (`✂️ Chunk Splitter` → `🧵 Chunk Merge`).
3. Splits **H3 AV LATENT** (`NestedTensor` video+audio, or plain 5D video) on the keyframe grid (`🧬 Latent Chunk Splitter` → `🧬 Latent Chunk Merge`).
4. Cuts **AUDIO** on the same timeline (`🔊 Audio Chunk Splitter` → `🎚️ Audio Chunk Merge`).
5. Diagnoses NestedTensor / `5k+2` / phase (`🔎 Latent Info`).
6. Grabs the last N frames as a reference (`⏮️ Last Frames`).

---

## Nodes

| Node | Class | What it is for |
|---|---|---|
| 🎬 MMH3 Frame Calculator | `MMH3_FrameCalculator` | Real H3 frame count for a requested duration (`17k+5` @24fps) |
| ✂️ MMH3 Chunk Splitter (Seamless) | `MMH3_ChunkSplitter` | Pixel path: cut `IMAGE` into 2–5 overlapping chunks |
| 🧵 MMH3 Chunk Merge (Seamless) | `MMH3_ChunkMerge` | Pixel path: strip pad, crossfade overlap, exact source length |
| 🧬 MMH3 Latent Chunk Splitter | `MMH3_LatentChunkSplitter` | Latent path: unbind NestedTensor, snap to keyframes, pad `5k+2` |
| 🧬 MMH3 Latent Chunk Merge | `MMH3_LatentChunkMerge` | Latent path: causal stitch, one NestedTensor, **one** `VAEDecode` |
| 🔎 MMH3 Latent Info | `MMH3_LatentInfo` | nested?, shapes, T, `5k+2`, pixel-frame estimate, audio match |
| 🔊 MMH3 Audio Chunk Splitter | `MMH3_AudioChunkSplitter` | Waveform cut on the same plan as video/latent |
| 🎚️ MMH3 Audio Chunk Merge | `MMH3_AudioChunkMerge` | Sample-accurate audio restitch (only if the 2nd pass mutates audio) |
| ⏮️ MMH3 Last Frames | `MMH3_LastFrames` | Last N frames → next H3 generation as reference |

Defaults that matter: `align_h3_grid=ON`, latent `overlap=5`, latent pad `5n+2`, Merge `blend_mode=causal`. Do **not** `VAEDecode` chunks separately — Merge first, then one decode.

---

# Русская документация

Ноды для ComfyUI, которые берут готовый ролик MiniMax H3 (пиксели или
латент — на выбор) и его звуковую дорожку, режут на 2–5 кусков с
нахлёстом для параллельного апскейла и второго прохода семплером, а
потом бесшовно склеивают всё обратно — число кадров (пиксельных,
латентных или сэмплов звука) на выходе **точно** совпадает с
оригиналом.

---

## Почему мерцало (и что исправлено)

Если вы декодировали куски **напрямую** с `🧬 MMH3 Latent Chunk Splitter`
и первый кусок был нормальный, а дальше шло мерцание/пульс — это не
Merge. Это два бага старого сплиттера, оба уже на нарезке:

### 1. H3 — это NestedTensor, не плоский 5D

`latent["samples"]` у MiniMax H3 — пара:

```
video  [B, 24, T,  H/16, W/16]     ось времени = dim 2
audio  [B, 32, 2,  Ta]             ось времени = dim 3
```

Старый код делал `samples[:, :, start:end]`. Этот же слайс падал на
**аудио dim 2** (стерео, размер 2): кусок 1 случайно забирал оба канала
(`0:T` клипается в `0:2`), кусок 2 резал мимо и получал пустое/битое
аудио. Видео при этом ещё и стартовало со сдвигом фазы (см. ниже).

Теперь: `unbind` → режем видео по dim 2, аудио по dim 3 через покрытие
кадров `(1,4,4,4,4)` → `NestedTensor` обратно.

### 2. Сетка keyframe `(1, 4, 4, 4, 4)` / `5k+2`

H3-VisualVAE — **каузальный** f16t4d24. Токен 0 каждого 5-токенного
блока — это 1-кадровый **keyframe**, остальные четыре токена кроют по
4 пиксельных кадра:

```
пиксельные кадры  = 17k + 5     (10с → 243, 15с → 362)
латент-токены     =  5k + 2     (10с →  72, 15с → 107)
```

VAEDecode внутри себя режет ролик 17-кадровыми каузальными окнами,
считая что вход **начинается с keyframe**. Старый сплиттер с
`overlap=2` на ролике T=72 давал куску 2 старт на токене 34
(`34 % 5 = 4`) — VAE принимал 4-кадровый residual за 1-кадровый
keyframe и с этого места **фаза (1,4,4,4,4) съезжала**. Отсюда пульс
с периодом ~17 кадров на всём, что после первого куска.

Даже split→merge→один decode мерцал, если по пути куски независимо
прогонялись семплером/декодом в этой сдвинутой фазе.

Теперь (дефолты):

| | Было | Стало |
|---|---|---|
| `overlap_frames` | 2 (не кратно 5) | **5** (один 17-кадровый блок), округляется до ×5 |
| `pad_multiple` / `pad_remainder` | 1 / 0 | **5 / 2** (каждый кусок — валидный клип `5k+2`) |
| старт куска | где придётся | **token % 5 == 0** (keyframe) |
| `blend_mode` в Merge | `smoothstep` | **`causal`** — хвост раннего куска, warmup позднего выбрасывается |
| NestedTensor | резался как 5D | unbind / slice по своим осям / bind |

`align_h3_grid=true` (по умолчанию) включает сетку. Выключить можно,
но тогда мерцание вернётся.

Старый overlap `2` **не** даёт нулевой нахлёст: планировщик округляет
значения `(0, 5]` **вверх** до 5, чтобы leftover-графы не получили
hard-cut на keyframe.

### Как проверять

1. Повесьте `🔎 MMH3 Latent Info` на исходный латент и на каждый кусок.
   В `info` должно быть `5k+2=YES` и `T%5=0` на старте куска (смотрите
   plan/info сплиттера: `keyframe OK`).
2. **Правильный decode:** Splitter → (опционально 2-й проход) →
   **Latent Chunk Merge** → **один** `VAEDecode` на весь ролик.
3. Decode кусков по отдельности допустим только после этого фикса
   (каждый кусок on-grid) — но всё равно хуже, чем один decode после
   Merge: каузальный VAE на старте куска 2 не имеет истории. Нахлёст
   в 5 токенов это прячет.

---

## Содержание

- [Почему мерцало (и что исправлено)](#почему-мерцало-и-что-исправлено)
- [Два пути: пиксельный и латентный](#два-пути-пиксельный-и-латентный)
- [Установка](#установка)
- [Ноды — входы, выходы, дефолты](#ноды--входы-выходы-дефолты)
- [Пошаговая сборка графа — пиксельный путь](#пошаговая-сборка-графа--пиксельный-путь)
- [Пошаговая сборка графа — латентный путь](#пошаговая-сборка-графа--латентный-путь)
- [Рекомендуемые настройки](#рекомендуемые-настройки)
- [Формат `plan` (JSON)](#формат-plan-json)
- [Частые ошибки](#частые-ошибки)
- [Тесты](#тесты)
- [Откуда взяты формулы](#откуда-взяты-формулы)
- [Changelog](#changelog)

---

## Два пути: пиксельный и латентный

| | Пиксельный путь (`Chunk Splitter/Merge`, IMAGE) | Латентный путь (`Latent Chunk Splitter/Merge`, LATENT) |
|---|---|---|
| Когда использовать | Апскейл — пиксельный (RTX VSR, TensorRT, любой upscale-by-model) | Апскейл — узел **Latent Upscale**, работающий прямо в латент-пространстве |
| VAEEncode/VAEDecode | На каждый кусок отдельно (encode перед семплером, decode после) | **Не нужен вообще** на кусок — один VAEDecode в самом конце всего пайплайна |
| overlap считается в | пиксельных кадрах | латент-токенах (кратно 5; 5 токенов = 17 пиксельных кадров ≈ 0.7 с @24fps) |
| `flow_align` blend | есть (экспериментально, нужен `opencv-python`) | нет; вместо него `causal` |
| Звук NestedTensor | не трогается (это waveform-AUDIO) | режется вместе с видео внутри Latent Splitter |

Оба пути используют один и тот же `🔊 MMH3 Audio Chunk Splitter` для
**waveform**-звука — разбивка считается пропорционально длительности. Можно
скормить `plan` из **любого** из двух Splitter'ов. Это **не** аудио-половина
NestedTensor: та едет вместе с видео внутри латентного пути.

Экономия латентного пути относительно пиксельного: **N encode + N decode → 0 encode + 1 decode**.

---

## Установка

```
cd ComfyUI/custom_nodes
git clone https://github.com/zeroleloushe/ComfyUI-MiniMaxSeamlessChunks
```

Или **ComfyUI Manager → Install via Git URL** → тот же адрес.

Обновление уже установленного пака:

```
cd ComfyUI/custom_nodes/ComfyUI-MiniMaxSeamlessChunks
git pull
```

Рестарт ComfyUI, затем **Ctrl+F5** (чтобы подтянулся `web/theme.js`).

Дерево после установки:

```
ComfyUI/custom_nodes/ComfyUI-MiniMaxSeamlessChunks/
├── __init__.py          # NODE_CLASS_MAPPINGS + WEB_DIRECTORY
├── nodes.py             # все 9 нод, включая Latent Splitter
├── plan.py              # 17k+5 / 5k+2 / plan_chunks / plan_chunks_h3
├── av_latent.py         # unbind NestedTensor, slice по своим осям, bind
├── web/theme.js         # фиолетовая тема + бейдж «MMH3 · seamless»
├── tests/               # сетка + NestedTensor roundtrip (без ComfyUI)
├── pyproject.toml
├── LICENSE              # MIT
└── README.md
```

Зависимостей кроме `torch` (уже есть в ComfyUI) — никаких.
`opencv-python` (`cv2`) нужен **только** для `blend_mode=flow_align` на
пиксельном Merge; без него этот режим тихо откатывается на `smoothstep`.

Python ≥ 3.10.

---

## Ноды — входы, выходы, дефолты

Все — в категории **MiniMax H3 → Seamless Chunks**. Фиолетовая тема и
уголок `MMH3 · seamless` рисуются из `web/theme.js`.

### 🎬 MMH3 Frame Calculator

Считает **реальное** число кадров MiniMax H3 для запрошенной длительности
(квантование `17k+5` при 24 fps).

| Вход | Тип | Дефолт | Смысл |
|---|---|---|---|
| `requested_seconds` | FLOAT 1–15 | `10.0` | Как вы вводите duration в генераторе H3 |
| `fps` | INT | `24` | Не менять, если явно не знаете другой fps модели |
| `block` / `remainder` | INT | `17` / `5` | Формула кадров. Не менять |

| Выход | Смысл |
|---|---|
| `frames` | Точное число кадров, которое отдаст H3 (10 с → **243**, 15 с → **362**) |
| `seconds` | `frames / fps` — можно подать обратно в duration генератора |
| `fps` | Проброс |

Поставьте **до** генерации, чтобы знать длину заранее, и/или **после**,
чтобы проверить скачанный ролик.

---

### ✂️ MMH3 Chunk Splitter (Seamless) — пиксельный путь

Режет `IMAGE` на 2–5 кусков с нахлёстом в **пиксельных** кадрах. Каждый
кусок дописывается дублями последнего кадра до формулы второго семплера
(`4n+1` по умолчанию — типичный temporal compression ×4).

| Вход | Дефолт | Смысл |
|---|---|---|
| `images` | — | Весь декодированный ролик (например VHS Load Video) |
| `num_chunks` | `4` (2–5) | Сколько кусков. Столько же `chunk_N` подключить дальше |
| `overlap_frames` | `8` | Кадры нахлёста. 8 @24fps ≈ ⅓ с. Для быстрого движения 12–16 |
| `pad_multiple` / `pad_remainder` | `4` / `1` | Формула входа 2-го семплера. `0` = не паддить |

| Выход | Смысл |
|---|---|
| `chunk_1..5` | Обычный `IMAGE`. Неподключённые слоты можно оставить пустыми |
| `plan` | JSON-карта. **Обязательно** довести до Merge (и до Audio Splitter) |
| `info` | Что и где задублировано при паддинге |

Неподключённые `chunk_N` всё равно заполняются крошечным dummy (`images[:1]`),
чтобы фиксированная арность ComfyUI не падала.

---

### 🧵 MMH3 Chunk Merge (Seamless) — пиксельный путь

Снимает pad, кроссфейдит нахлёст, длина на выходе = `plan.total_frames`.

| Вход | Дефолт | Смысл |
|---|---|---|
| `plan` | forceInput | Тот же JSON из Splitter |
| `chunk_1` | required | Обработанный кусок 1 |
| `chunk_2..5` | optional | Подключать ровно `num_chunks` штук |
| `blend_mode` | `smoothstep` | см. ниже |

`blend_mode`:

| Режим | Когда |
|---|---|
| `smoothstep` | По умолчанию. Плавный кроссфейд, мало мерцания |
| `flow_align` | Экспериментально: Farneback optical flow выравнивает микросдвиг между двумя независимыми рендерами **одного** timestamp, потом smoothstep. На синтетических тестах пакета **не** выигрывает у plain smoothstep (оценка потока сама шумит). Нужен `opencv-python`; без него — тихий fallback на smoothstep |
| `linear` | Линейный кроссфейд |
| `equal_energy` | Сохраняет яркость на контрастных стыках |
| `hard_cut` | Разрез посередине нахлёста, без смешивания |

Все куски должны быть **одного** H×W. Если апскейлер отработал не на всех —
ресайзните до Merge.

---

### 🧬 MMH3 Latent Chunk Splitter — латентный путь

Режет H3 AV-LATENT (NestedTensor видео+аудио **или** плоский 5D) на 2–5
кусков **по keyframe-сетке**. 4D image-latent не подходит.

| Вход | Дефолт | Смысл |
|---|---|---|
| `latent` | — | Весь AV-LATENT после H3 / Latent Upscale |
| `num_chunks` | `2` (2–5) | На сколько резать |
| `overlap_frames` | **`5`** | Нахлёст в **латент-токенах**, кратно 5. `2` — это и был баг |
| `align_h3_grid` | **`true`** | Старт с keyframe, overlap ×5, pad `5k+2`. Выключать не надо |
| `pad_multiple` / `pad_remainder` | `5` / `2` | Формула длины куска. Для H3 это `5n+2` |

| Выход | Смысл |
|---|---|
| `latent_chunk_1..5` | `LATENT` (NestedTensor, если вход был nested). Прямо в KSampler |
| `plan` | JSON в latent tokens. Довести до Latent Chunk Merge. Подойдёт и Audio Splitter |
| `info` | nested?, T, `5k+2`, фаза каждого куска (`keyframe OK` / `PHASE n`), паддинг |

`noise_mask` исходного латента **не** режется и не переносится в куски —
если второй проход его требует, добавьте отдельно.

Декодировать лучше **один раз** после Merge. Отдельный VAEDecode куска 2+
без keyframe-выравнивания и есть 17-кадровое мерцание.

---

### 🧬 MMH3 Latent Chunk Merge — латентный путь

Собирает куски в один NestedTensor точной исходной длины. Декодируйте
**одним** `VAEDecode`.

| Вход | Дефолт | Смысл |
|---|---|---|
| `plan` | forceInput | Из Latent Chunk Splitter |
| `latent_chunk_1` | required | Обработанный кусок 1 |
| `latent_chunk_2..5` | optional | Ровно `num_chunks` штук |
| `blend_mode` | **`causal`** | см. ниже |

| Режим | Поведение |
|---|---|
| **`causal`** | Хвост **раннего** куска, warmup позднего выбрасывается. Identity на необработанном сплите. Единственный режим, который не усредняет две разные фазы VAE после независимого 2-го прохода |
| `smoothstep` / `linear` / `equal_energy` | Кроссфейд в латенте. Имеет смысл, только если оба куска прогнаны одним и тем же 2-м проходом **на выровненной сетке** |
| `hard_cut` | Разрез посередине нахлёста |

B/C/H/W всех кусков должны совпасть (один коэффициент Latent Upscale на всех).

---

### 🔎 MMH3 Latent Info

Диагностика. Вешайте на исходник и на куски, если снова что-то мигает.

| Выход | Смысл |
|---|---|
| `info` | nested yes/no, `video [B,C,T,H,W]`, `5k+2=YES/NO`, `T%5`, `pixel_frames≈…`, audio shape, предупреждения (C≠24, Ta не бьётся с `frames×5/3`) |
| `latent_tokens` | T видео-латента (dim 2) |
| `pixel_frames_est` | Пиксельные кадры по сетке `(1,4,4,4,4)` для этого T |

---

### 🔊 MMH3 Audio Chunk Splitter / 🎚️ MMH3 Audio Chunk Merge

Синхронный **waveform** по тем же границам, что видео/латент (тот же
нахлёст). Считает sample-accurate `audio_plan` с накопительным округлением,
чтобы core-сегменты всегда суммировались в исходное число сэмплов.

Нужно, если второй проход зависит от звука (липсинк / audio-conditioned
sampler). Если звук **не** меняется — Audio Merge не нужен: замьюксируйте
исходную полную дорожку поверх готового видео.

Audio Merge `blend_mode`: `equal_power` (дефолт, без провала громкости) /
`linear`.

---

### ⏮️ MMH3 Last Frames

Последние N кадров ролика — в референс следующей генерации H3 или во все
ветки второго прохода, чтобы держать единый стиль.

| Вход | Дефолт |
|---|---|
| `images` | IMAGE-батч |
| `count` | `1` (1–64) |

---

## Пошаговая сборка графа — пиксельный путь

```
[MiniMax H3] → IMAGE
   ▼
[✂️ Chunk Splitter] num_chunks=4, overlap=8
   │ chunk_1..4        plan
   ▼                    │
[Апскейлер] x4           │
   ▼                    │
[VAEEncode] x4            │
   ▼                    │
[KSampler] x4 ← общий seed/prompt
   ▼                    │
[VAEDecode] x4            │
   └──────────┬──────────┘
              ▼
      [🧵 Chunk Merge]  blend_mode=smoothstep
              │ IMAGE
              ▼
      [Save/Combine Video]
```

Звук (если 2-й проход его ест): тот же `plan` → `🔊 Audio Chunk Splitter` →
в каждую ветку семплера → `🎚️ Audio Chunk Merge`.

---

## Пошаговая сборка графа — латентный путь

```
[MiniMax H3] → NestedTensor LATENT
   ▼
[Latent Upscale] (ОДИН раз, весь ролик, спатиальный)
   ▼
[🔎 Latent Info]          ← T=72, 5k+2=YES для 10с
   ▼
[🧬 Latent Chunk Splitter] num_chunks=2, overlap=5, align_h3_grid=ON
   │ latent_chunk_1..N        plan
   ▼                           │
[KSampler] xN ← общий seed     │   (VAEEncode НЕ нужен)
   ▼                           │
   └──────────┬────────────────┘
              ▼
      [🧬 Latent Chunk Merge]  blend_mode=causal
              │ LATENT
              ▼
      [VAEDecode] (ОДИН раз)
              ▼
      [Save/Combine Video]
```

Для 10-секундного ролика (T=72) при `num_chunks=2, overlap=5`:

- кусок 1 стартует на токене 0 (keyframe)
- кусок 2 стартует на токене, кратном 5 (не 34)
- оба паддятся до `5k+2`
- Merge `causal` отдаёт ровно 72 токена → 243 пиксельных кадра

---

## Рекомендуемые настройки

| Сценарий | num_chunks | overlap | pad | blend |
|---|---|---|---|---|
| 10 с, латентный 2-й проход | 2 | 5 токенов | 5 / 2 | `causal` |
| 15 с, латентный 2-й проход | 2–4 | 5 | 5 / 2 | `causal` |
| Быстрое движение в кадре, пиксельный апскейл | 4 | 12–16 кадров | 4 / 1 | `smoothstep` |
| Спокойный кадр, пиксельный апскейл | 4 | 8 | 4 / 1 | `smoothstep` |
| Липсинк во 2-м проходе | те же, плюс Audio Splitter/Merge | — | — | audio: `equal_power` |

Не ставьте `num_chunks` так, чтобы `core_len ≤ overlap` — планировщик
бросит `ValueError`. Для T=72 максимум 4 куска при overlap=5; 5 кусков
проходят на T=107 (15 с).

---

## Формат `plan` (JSON)

Splitter пишет, Merge читает. Не редактируйте руками, если не понимаете
поля. Ключевые:

```json
{
  "total_frames": 72,
  "num_chunks": 2,
  "overlap": 5,
  "overlap_snapped": 5,
  "pad_multiple": 5,
  "pad_remainder": 2,
  "unit": "latent_frames",
  "align_h3_grid": true,
  "was_nested": true,
  "source_on_grid": true,
  "pixel_frames_est": 243,
  "chunks": [
    {
      "index": 0,
      "core_start": 0,
      "core_len": 35,
      "left_ov": 0,
      "right_ov": 5,
      "raw_start": 0,
      "raw_end": 40,
      "raw_len": 40,
      "pad": 2,
      "final_len": 42,
      "start_phase": 0,
      "on_keyframe": true,
      "final_on_grid": true
    }
  ]
}
```

- `core_*` — уникальный кусок таймлайна. Сумма `core_len` = `total_frames`.
- `left_ov` / `right_ov` — контекст с соседа (нахлёст).
- `raw_*` — окно нарезки, включая нахлёст.
- `pad` / `final_len` — дописка до формулы семплера; Merge снимает pad
  (`tensor[:raw_len]`), потом кроссфейдит `right_ov`.
- Пиксельный Splitter пишет тот же каркас без H3-полей (`unit` нет,
  `align_h3_grid` нет).
- Audio Splitter пишет отдельный `audio_plan` в сэмплах
  (`core_len_samples`, `left_ov_samples`, …).

---

## Частые ошибки

| Сообщение / симптом | Причина | Что делать |
|---|---|---|
| Мерцание со 2-го куска, первый нормальный | Старый сплиттер / `align_h3_grid=off` / overlap не кратен 5 / decode кусков по отдельности | `git pull`, overlap=5, Merge → один VAEDecode |
| `expected a 5D video latent` | На вход попал 4D image-latent или сломанный NestedTensor | Это выход H3 / видео-VAE, не SD-картинка. Повесить Latent Info |
| `latent_chunk_N shape does not match` | Апскейл по-разному на разных кусках | Один коэффициент Latent Upscale на всех |
| `WARNING: reconstructed T != source` / `not 5k+2` | Семплер сменил длину куска | Не менять число токенов во 2-м проходе; pad `5n+2` |
| `chunk_N is required but was not connected` | `num_chunks` и число входов Merge не совпадают | Свести числа |
| `overlap too large for core length` | Слишком большой нахлёст на коротком ролике | Уменьшить overlap или `num_chunks` |
| `total_tokens too short to place N keyframe-aligned chunks` | Ролик короче `N×5` токенов тела | Меньше кусков |
| Не появилось оформление у нод | Кэш `theme.js` | Ctrl+F5 |
| `flow_align` без выигрыша / тихий fallback | Нет `opencv-python`, или Farneback шумит на гладких областях | Оставить `smoothstep` |

---

## Тесты

Без ComfyUI. Из корня пака:

```
python -m unittest tests.test_plan tests.test_roundtrip -v
```

- `tests/test_plan.py` — сетка `17k+5` / `5k+2`, покрытие `(1,4,4,4,4)`,
  `plan_chunks_h3` всегда стартует с keyframe, старый overlap=2 даёт
  phase 4 (репродукция бага), audio range идёт по пиксельным кадрам,
  а не пропорционально индексу токена.
- `tests/test_roundtrip.py` — нужен `torch` (+ `numpy`). Naive
  `samples[:,:,s:e]` на NestedTensor ломает аудио dim 2 и фазу;
  unbind+`slice_av_window` сохраняет stereo и keyframe; split→merge
  `causal` — identity на необработанном видео.

---

## Откуда взяты формулы

- Квантование кадров MiniMax H3 (`17k+5` @24fps) — 10с → 243, 15с → 362.
- VisualVAE `f16t4d24`, каузальный, `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`,
  латент `5k+2` — MiniMax tech report + community pack
  `bbaudio-2025/Comfyui-MiniMax-H3-LatentSplit` (`snap_frame_boundary(phase=5)`)
  и `ckinpdx/ComfyUI-MMH3Tools` («concatenation of two on-grid chunks is
  off-grid; the VAE's 17-frame causal chunking misaligns and the second
  half pulses»).
- NestedTensor `(video, audio)` — `comfy.nested_tensor.NestedTensor`.
- Аудио-токены `round(frames * 5/3)` (40 Hz / 24 fps).

---

## Changelog

### 1.1.0

- **Latent Chunk Splitter / Merge / Info** грузятся всегда (больше не
  optional import). Пиксельный Splitter/Merge, Audio Splitter/Merge и
  Frame Calculator — в том же `nodes.py`.
- Unbind NestedTensor: видео режется по dim 2, аудио по dim 3 через
  покрытие `(1,4,4,4,4)`.
- `plan_chunks_h3`: старт куска на keyframe (`token % 5 == 0`), overlap
  снапится к ×5 (старый дефолт 2 → 5, не в 0), pad каждого куска до `5k+2`.
- Merge по умолчанию `causal`.
- `🔎 Latent Info` и тесты сетки / NestedTensor-бага.
- Этот README.

### 1.0.0

- Первая публикация латентного сплита на сетке `5k+2`.

---

MIT © 2026 Zero Leloushe
