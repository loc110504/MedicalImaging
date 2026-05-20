# Spec: EUGIS-style Uncertainty-Guided Point Click Generation + Visualization

## 1. Goal

Implement a standalone inference/visualization extension for the existing script that already produces:

- predicted segmentation mask `pred_256`
- ground-truth mask `label_256`
- uncertainty maps:
  - `maxprob_unc`
  - `entropy_unc`
  - `evidential_unc`

The new code must generate **point clicks** following the EUGIS idea: select point prompts from pixels with the highest uncertainty scores. This task only covers point-click generation and visualization. **Do not implement prompt encoder or interactive refinement yet.**

The expected output is a set of visualization images showing where the generated clicks fall on:

1. original image
2. predicted mask
3. ground-truth mask
4. selected uncertainty map
5. optional overlay view

The number of clicks must be configurable from CLI.

---

## 2. Background from EUGIS

EUGIS uses a two-stage design:

- Stage I generates an evidential uncertainty map.
- Stage II uses the uncertainty map to simulate point prompts.

For prompt simulation, EUGIS selects the top uncertain pixels from the current uncertainty map. In the full paper, `k` is the number of point prompts sampled at a single interaction step, and `M` is the number of interaction steps. In the main 1/3/5-click protocol, they use `k = 1` and control the click budget with `M ∈ {1, 3, 5}`. The implementation requested here can simplify this by directly generating `num_clicks` points from the selected uncertainty map.

Important: this implementation should generate only the coordinates and optional polarity labels of clicks. It should not feed them to a prompt encoder.

---

## 3. Inputs

Reuse the current script inputs and add these CLI arguments:

```bash
--num_clicks 5
--uncertainty_type evidential
--click_selection topk
--min_click_distance 8
--click_polarity gt
--foreground_classes 1 2 3
--save_click_json
--save_click_csv
--show_click_index
```

### 3.1 Required new arguments

#### `--num_clicks`

Type: `int`  
Default: `5`  
Meaning: number of clicks to generate per slice.

#### `--uncertainty_type`

Type: `str`  
Choices:

- `evidential`
- `entropy`
- `maxprob`

Default: `evidential`

Meaning: choose which uncertainty map to use for point selection.

#### `--click_selection`

Type: `str`  
Choices:

- `topk`
- `topk_nms`
- `error_aware_oracle`

Default: `topk_nms`

Meaning:

- `topk`: select the globally highest uncertainty pixels.
- `topk_nms`: select high-uncertainty pixels while enforcing a minimum distance between selected clicks.
- `error_aware_oracle`: optional debugging/oracle mode using GT and prediction to prefer false-negative/false-positive error regions. This is not the pure EUGIS rule, but useful to inspect positive/negative click behavior.

#### `--min_click_distance`

Type: `int`  
Default: `8`  
Meaning: minimum Euclidean pixel distance between two selected clicks when using `topk_nms`.

#### `--click_polarity`

Type: `str`  
Choices:

- `none`
- `gt`
- `pred`
- `error_type`

Default: `gt`

Meaning:

- `none`: generate coordinates only, no positive/negative label.
- `gt`: label point as positive if it falls inside GT foreground, else negative.
- `pred`: label point as positive if it falls inside predicted foreground, else negative. Useful for deployment-like simulation without GT.
- `error_type`: label point based on error type: false negative → positive, false positive → negative. If point is not in an error region, fall back to GT label.

#### `--foreground_classes`

Type: list of ints  
Default: all nonzero classes, i.e. `[1, 2, 3]` for ACDC when `num_classes=4`.

Meaning: classes considered foreground when assigning positive/negative labels.

---

## 4. Output files

For each visualized slice, save the original existing figure and one new figure:

```text
{case_name}_slice{slice_idx}_eugis_clicks_{uncertainty_type}_{num_clicks}.png
```

Optional metadata outputs:

```text
{case_name}_slice{slice_idx}_eugis_clicks_{uncertainty_type}_{num_clicks}.json
{case_name}_slice{slice_idx}_eugis_clicks_{uncertainty_type}_{num_clicks}.csv
```

Each click metadata record should contain:

```json
{
  "case_name": "patient001",
  "slice_idx": 5,
  "click_id": 1,
  "x": 123,
  "y": 87,
  "uncertainty": 0.9421,
  "label": 1,
  "label_name": "positive",
  "source": "evidential",
  "selection_method": "topk_nms"
}
```

Coordinate convention:

- `x`: column index
- `y`: row index
- Coordinates are in the resized 256×256 image space.

---

## 5. Core functions to implement

Add the following functions to the script or to a helper module, for example:

```text
code/utils/eugis_clicks.py
```

### 5.1 `get_selected_uncertainty_map(...)`

Signature:

```python
def get_selected_uncertainty_map(
    uncertainty_type: str,
    maxprob_unc: np.ndarray,
    entropy_unc: np.ndarray,
    evidential_unc: np.ndarray,
) -> np.ndarray:
    ...
```

Behavior:

- Return the chosen uncertainty map.
- Validate `uncertainty_type`.
- Ensure returned array is `float32`, shape `[H, W]`, normalized to `[0, 1]`.

---

### 5.2 `make_foreground_mask(...)`

Signature:

```python
def make_foreground_mask(mask: np.ndarray, foreground_classes: list[int]) -> np.ndarray:
    ...
```

Behavior:

- Return a boolean foreground mask.
- For binary segmentation, foreground is `mask > 0`.
- For multiclass segmentation, foreground is `np.isin(mask, foreground_classes)`.

---

### 5.3 `select_topk_uncertain_points(...)`

Signature:

```python
def select_topk_uncertain_points(
    uncertainty: np.ndarray,
    num_clicks: int,
    exclude_mask: np.ndarray | None = None,
) -> list[dict]:
    ...
```

Behavior:

- Flatten uncertainty map.
- Sort pixels by uncertainty descending.
- Optionally ignore pixels where `exclude_mask == True`.
- Select the first `num_clicks` pixels.
- Return list of dictionaries:

```python
[
    {"x": int(x), "y": int(y), "uncertainty": float(value)},
    ...
]
```

Important:

- Avoid duplicate points.
- If there are fewer valid pixels than `num_clicks`, return as many as possible and print a warning.

---

### 5.4 `select_topk_uncertain_points_nms(...)`

Signature:

```python
def select_topk_uncertain_points_nms(
    uncertainty: np.ndarray,
    num_clicks: int,
    min_distance: int = 8,
    exclude_mask: np.ndarray | None = None,
) -> list[dict]:
    ...
```

Behavior:

- Sort all pixels by uncertainty descending.
- Iteratively select the next highest-uncertainty pixel only if it is at least `min_distance` pixels away from all previously selected clicks.
- This avoids multiple clicks collapsing into the same small high-uncertainty blob.
- Return same format as `select_topk_uncertain_points`.

Distance:

```python
distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
```

Acceptance rule:

```python
distance >= min_distance
```

Recommended default: use this method for visualization because pure top-k often produces visually redundant clustered clicks.

---

### 5.5 `assign_click_polarity(...)`

Signature:

```python
def assign_click_polarity(
    clicks: list[dict],
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    foreground_classes: list[int],
    mode: str = "gt",
) -> list[dict]:
    ...
```

Behavior:

Add these fields to each click:

```python
"label": 1 or 0 or None
"label_name": "positive" or "negative" or "unknown"
```

Modes:

#### `none`

```python
label = None
label_name = "unknown"
```

#### `gt`

Use ground-truth foreground membership:

```python
label = 1 if gt_fg[y, x] else 0
```

This is recommended for training visualization because GT is available.

#### `pred`

Use prediction foreground membership:

```python
label = 1 if pred_fg[y, x] else 0
```

This simulates deployment behavior without GT, but can be wrong if prediction is wrong.

#### `error_type`

Use error type:

```python
fn = gt_fg & (~pred_fg)      # missed foreground
fp = (~gt_fg) & pred_fg      # false foreground

if fn[y, x]:
    label = 1  # positive click: add missed object region
elif fp[y, x]:
    label = 0  # negative click: remove false object region
else:
    label = 1 if gt_fg[y, x] else 0
```

This mode is not explicitly described in EUGIS, but is useful as an oracle diagnostic for positive/negative prompt logic.

---

### 5.6 `select_error_aware_oracle_points(...)`

Optional but recommended.

Signature:

```python
def select_error_aware_oracle_points(
    uncertainty: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    num_clicks: int,
    foreground_classes: list[int],
    min_distance: int = 8,
) -> list[dict]:
    ...
```

Behavior:

- Compute foreground masks for GT and prediction.
- Compute false-negative and false-positive regions:

```python
fn = gt_fg & (~pred_fg)
fp = (~gt_fg) & pred_fg
```

- Create score maps:

```python
score_pos = uncertainty * fn
score_neg = uncertainty * fp
```

- Iteratively select the highest available point from either `score_pos` or `score_neg`.
- If selected from `score_pos`, label is positive.
- If selected from `score_neg`, label is negative.
- Enforce NMS distance.
- If no error-region score remains, fall back to global uncertainty top-k-NMS and assign polarity by GT.

This mode is good for debugging whether clicks really target meaningful correction areas.

---

## 6. Visualization requirements

Add a new function:

```python
def save_click_figure(
    case_name: str,
    slice_idx: int,
    image_256: np.ndarray,
    label_256: np.ndarray,
    pred_256: np.ndarray,
    uncertainty: np.ndarray,
    clicks: list[dict],
    uncertainty_type: str,
    output_dir: str,
    show_click_index: bool = True,
) -> str:
    ...
```

### 6.1 Figure layout

Create a figure with 1 row and 5 columns:

1. `Image + Clicks`
2. `GT + Clicks`
3. `Pred + Clicks`
4. `{UncertaintyType} Unc + Clicks`
5. `Image + Pred Overlay + Clicks`

Recommended size:

```python
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
```

### 6.2 Click colors and markers

Use consistent visual convention:

- Positive click:
  - marker: circle `o`
  - color: lime/green
  - edgecolor: black
- Negative click:
  - marker: `x`
  - color: red
- Unknown/no polarity:
  - marker: star `*`
  - color: yellow
  - edgecolor: black

Recommended drawing:

```python
if label == 1:
    ax.scatter(x, y, c="lime", marker="o", s=70, edgecolors="black", linewidths=1.0)
elif label == 0:
    ax.scatter(x, y, c="red", marker="x", s=80, linewidths=2.0)
else:
    ax.scatter(x, y, c="yellow", marker="*", s=90, edgecolors="black", linewidths=1.0)
```

If `show_click_index=True`, draw click number near the point:

```python
ax.text(x + 3, y + 3, str(click_id), color="white", fontsize=8,
        bbox=dict(facecolor="black", alpha=0.5, pad=1))
```

### 6.3 Overlay requirements

For the `Image + Pred Overlay + Clicks` panel:

- Show normalized image in grayscale.
- Overlay predicted foreground mask with alpha around `0.35`.
- For multiclass prediction, either:
  - overlay all nonzero foreground as one mask, or
  - use the existing `seg_cmap` with alpha.

Simplest version:

```python
ax.imshow(normalize01(image_256), cmap="gray")
ax.imshow(pred_fg, cmap="Reds", alpha=0.35)
```

### 6.4 Figure title

Use a clear suptitle:

```text
{case_name} slice {slice_idx} | {num_clicks} EUGIS-style clicks | uncertainty={uncertainty_type} | polarity={click_polarity}
```

---

## 7. Main-loop integration

In the existing `main()` loop, after this block:

```python
pred_256, maxprob_unc, entropy_unc, evidential_unc = predict_and_uncertainty(
    net, x, args.num_classes
)
```

Add:

```python
selected_unc = get_selected_uncertainty_map(
    args.uncertainty_type,
    maxprob_unc,
    entropy_unc,
    evidential_unc,
)

if args.click_selection == "topk":
    clicks = select_topk_uncertain_points(
        selected_unc,
        args.num_clicks,
    )
elif args.click_selection == "topk_nms":
    clicks = select_topk_uncertain_points_nms(
        selected_unc,
        args.num_clicks,
        min_distance=args.min_click_distance,
    )
elif args.click_selection == "error_aware_oracle":
    clicks = select_error_aware_oracle_points(
        selected_unc,
        label_256,
        pred_256,
        args.num_clicks,
        foreground_classes=args.foreground_classes,
        min_distance=args.min_click_distance,
    )
else:
    raise ValueError(f"Unknown click_selection: {args.click_selection}")

if args.click_selection != "error_aware_oracle":
    clicks = assign_click_polarity(
        clicks,
        gt_mask=label_256,
        pred_mask=pred_256,
        foreground_classes=args.foreground_classes,
        mode=args.click_polarity,
    )

click_fig_path = save_click_figure(
    case_name,
    slice_idx,
    image_256,
    label_256,
    pred_256,
    selected_unc,
    clicks,
    args.uncertainty_type,
    args.output_dir,
    show_click_index=args.show_click_index,
)
print("saved clicks:", click_fig_path)
```

---

## 8. Expected CLI examples

### 8.1 EUGIS-like top uncertainty clicks using evidential uncertainty

```bash
python infer_uncertainty_clicks.py \
  --root_path ../../data/ACDC \
  --model unet_hl \
  --num_classes 4 \
  --checkpoint ../../checkpoints/ACDC_EvidentialFullMask/unet_hl_best_model.pth \
  --num_slices 5 \
  --num_clicks 5 \
  --uncertainty_type evidential \
  --click_selection topk_nms \
  --min_click_distance 8 \
  --click_polarity gt \
  --output_dir ../../code/test/eugis_click_vis
```

### 8.2 Compare entropy uncertainty clicks

```bash
python infer_uncertainty_clicks.py \
  --num_clicks 5 \
  --uncertainty_type entropy \
  --click_selection topk_nms \
  --click_polarity gt
```

### 8.3 Oracle debug mode using error regions

```bash
python infer_uncertainty_clicks.py \
  --num_clicks 5 \
  --uncertainty_type evidential \
  --click_selection error_aware_oracle \
  --min_click_distance 8
```

---

## 9. Acceptance criteria

The implementation is correct if:

1. Running the script produces the original uncertainty visualization as before.
2. Running the script also produces one click visualization per sampled slice.
3. `--num_clicks N` generates at most `N` clicks per slice.
4. `topk_nms` clicks are spatially separated by at least `--min_click_distance` pixels unless there are not enough valid candidates.
5. Positive/negative points are visually distinguishable.
6. The click metadata contains `x`, `y`, uncertainty value, and optional polarity label.
7. The code does not implement prompt encoder or mask refinement.
8. The implementation works for ACDC multiclass masks with foreground classes `[1, 2, 3]`.
9. The implementation does not require ground truth unless using `click_polarity=gt`, `click_polarity=error_type`, or `click_selection=error_aware_oracle`.

---

## 10. Important implementation notes

### 10.1 Coordinate order

Matplotlib uses:

```python
ax.scatter(x, y)
```

NumPy indexing uses:

```python
arr[y, x]
```

Be careful not to swap `x` and `y`.

### 10.2 Pure EUGIS vs debugging/oracle variants

The closest implementation to EUGIS is:

```bash
--click_selection topk_nms --uncertainty_type evidential
```

The following are extensions for analysis/debugging:

```bash
--click_polarity error_type
--click_selection error_aware_oracle
```

Do not describe these oracle/debug modes as exactly what EUGIS does.

### 10.3 Why NMS is recommended

Pure top-k may select many adjacent pixels from the same high-uncertainty boundary blob. This is technically consistent with top-k selection, but poor for visualization and prompt diversity. `topk_nms` keeps the uncertainty-guided idea while making clicks more spatially useful.

### 10.4 No prompt encoder yet

The output of this task should stop at:

```python
clicks = [{"x": ..., "y": ..., "label": ..., "uncertainty": ...}, ...]
```

Do not pass clicks into the network.
Do not change the segmentation prediction based on clicks.
Do not add iterative refinement yet.

---

## 11. Suggested final file names

If creating a new script:

```text
code/test/infer_eugis_clicks.py
```

If creating a helper module:

```text
code/utils/eugis_clicks.py
```

If modifying the current script, preserve existing behavior and only add new optional click-generation features.
