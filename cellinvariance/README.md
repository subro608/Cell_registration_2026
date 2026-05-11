# Cell Invariance — Cross-Modality Cell Re-Identification

Re-identify the same neuron across two GCaMP microscopy modalities — *in
vivo* (`invivo_raw`) and fixed *ex vivo* (`exvivo_registered`) — from a small
image patch around the cell. 115 paired landmark cells, frozen DINOv2-small
features, supervised-contrastive + Sinkhorn cross-modality alignment, no
test-time access to landmark coordinates.

![pipeline schematic](viz/schematic.png)

*Source volumes → 80 µm patches around 115 paired landmark cells →
`hard_intra` augmentations (×800/cell) → frozen DINOv2-small (3 scales) →
trained MLP 1152→256→64 → SupCon (intra-modality) + Sinkhorn (cross-modality)
→ trained 64-d embeddings (PCA-2D, same color = same cell, ● iv ◆ ex).*

## Headline

| Metric | Value |
|---|---|
| LOOCV R@1 (ex → iv) | **0.0887** |
| LOOCV R@5, cell-prototype | **0.2498** |
| MRR, cell-prototype | 0.1742 |
| Within-modality balanced knn accuracy | 0.9979 |

The within-modality features are nearly perfect; the cross-modality bridge is
the hard part.

## Method (w8)

1. Frozen `facebook/dinov2-small`, multi-scale CLS at xy ∈ {40, 80, 120} µm,
   z ∈ {12, 24, 36} µm → concatenated 1152-d feature.
2. 800 `hard_intra` (rotation-heavy) augmentations per cell, embedded once
   and cached.
3. MLP projector 1152 → 256 → 64 (L2-normalized), trained 200 epochs with
   supervised contrastive (τ = 0.1) + Sinkhorn cross-modality alignment
   (λ = 0.7). Batch 1024, AdamW, lr 1e-3.
4. Evaluation: per-cell leave-one-out, ex → iv top-K cosine retrieval.

See [`results/findings.md`](results/findings.md) for the full write-up.

## Quick start

```bash
uv sync
uv run python scripts/download_data.py             # one-time: pull the 3 dataset files
uv run python scripts/reproduce_w8.py --skip-train # regenerate viz from bundled projector (~10 min)
```

The bundled [`viz/dino_repr_interactive.html`](viz/dino_repr_interactive.html)
already shows the trained model. The third line re-derives it from
[`results/w8_projector.pt`](results/w8_projector.pt) by computing fresh DINO
features over the data you just downloaded.

Full retrain from scratch (first run takes 12–20 h on MPS to embed all
augmented patches; subsequent runs are ~30 min):

```bash
uv run python scripts/reproduce_w8.py
```

## Visualization

`viz/dino_repr_interactive.html` is a self-contained interactive HTML with
three coupled panels:

1. **Representation scatter** — PCA-2D of the 115 cells in the raw 1152-d
   DINO space, the 256-d MLP middle layer, and the 64-d MLP output. Both
   modalities (iv ●, ex ◆) overlaid; same color per cell. Augmentations as
   small translucent dots.
2. **Discriminability matrix** — 115×115 cosine similarity over the trained
   64-d representation.
3. **EX → IV contact sheet** — for every ex-vivo query, the top-5 in-vivo
   retrievals as image thumbnails, with the correct match highlighted when
   in the top-5.

## Repo layout

```
.
├── src/                 # core library (frozen DINO pipeline + supervised contrastive + sinkhorn)
│   ├── runner.py        # patch extraction, multi-scale DINOv2 feature pipeline
│   ├── contrastive.py   # MLPProjector + LOOCV training & evaluation
│   ├── augmentations.py # rotation augmentation presets
│   └── alignment.py     # Sinkhorn cross-modality alignment loss
├── scripts/
│   ├── reproduce_w8.py    # end-to-end orchestrator (train → install → regen viz)
│   ├── download_data.py   # fetch the 3 dataset files from Dropbox into data/
│   └── build_schematic.py # regenerate the README pipeline diagram
├── viz/
│   ├── generate_data.py             # build the data JSON from a trained projector
│   ├── generate_html.py             # render the data JSON to an interactive HTML
│   ├── dino_repr_interactive.html   # bundled visualization (15 MB)
│   └── schematic.png                # bundled pipeline diagram (~2 MB)
├── results/
│   ├── w8_projector.pt    # bundled trained MLP projector weights (1.2 MB)
│   ├── w8_results.json    # full LOOCV metrics
│   ├── w8_folds.jsonl     # per-fold breakdown
│   ├── w8_training.log    # original training log
│   └── findings.md        # detailed write-up
├── configs/
│   └── rotation_domain_invariance.yaml  # dataset + patch + rotation config (vendored)
├── data/                  # populated by scripts/download_data.py (gitignored)
├── pyproject.toml
└── README.md
```

## Data

The dataset config ships in [`configs/rotation_domain_invariance.yaml`](configs/rotation_domain_invariance.yaml)
and points at `./data/`. The three raw image / landmark files live on Dropbox
and are *not* committed:

| File | Purpose | Dropbox |
|---|---|---|
| `slice3_to_invivoLANDMARKS.json` | 115 paired landmark coordinates | [download](https://www.dropbox.com/scl/fi/ou6q1b2czrruynnm5kc6v/slice3_to_invivoLANDMARKS.json?rlkey=qr1cymvkm5tkcnk35uq9165rn&dl=1) |
| `zstack.tif` | in-vivo GCaMP volume | [download](https://www.dropbox.com/scl/fi/opfa4wetmw4w0tngr15qv/zstack.tif?rlkey=ddbopk6tumvi5phz9744xffhm&st=s1pn5ff4&dl=1) |
| `Sparrow_3_po_488_4x-registered.tif` | ex-vivo registered volume | [download](https://www.dropbox.com/scl/fi/sgviwbb64qypjrjw0xtyg/Sparrow_3_po_488_4x-registered.tif?rlkey=wds9m9a1lx7rs9ebax3ukc6df&dl=1) |

Fetch all three into `data/` in one go:

```bash
uv run python scripts/download_data.py
```

The script is idempotent — existing files are skipped. To put the data
elsewhere, set `CELLINVARIANCE_DATA_DIR=/path/to/data` before running any
script. The bundled `configs/rotation_domain_invariance.yaml` is preferred
over a cellfind checkout, but if you have one you can still point at it via
`CELLFIND_ROOT=/path/to/cellfind`.

## License

MIT — see [`LICENSE`](LICENSE).
