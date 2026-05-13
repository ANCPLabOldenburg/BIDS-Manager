# Screenshots checklist

Drop the captured images here under the exact filenames listed below. The
top-level `README.md` references them by relative path, so as soon as a
file lands, the broken-image icon on GitHub turns into the real shot.

Use **PNG** for stills, **GIF** or **MP4** for short walkthroughs. Keep each
file under **6 MB**. Capture on **dark theme** unless noted.

| # | File | What to capture | Size |
|---|------|-----------------|------|
| 1 | `hero.png` | Editor tab on a real dataset. BIDS tree on the left, NIfTI tri-view in the centre, validation pane on the right with a few severity chips. A short GIF (one drag of the crosshair, one click on a validation error) works even better than a still. | 1600 x 900 |
| 2 | `converter_view.png` | Converter tab full screen. Inspection table showing about 10 rows of a real multimodal scan, with 1 or 2 rows highlighted. | 1600 x 900 |
| 3 | `converter_running.png` | Converter mid-run. Progress chips lit up, BIDS preview pane filling, inspection table greyed out. | 1600 x 900 |
| 4 | `editor_sidecar.png` | Editor with `T1w.json` open in the sidecar form (BIDS view). At least 6 colour-coded fields visible (red REQUIRED, amber RECOMMENDED). | 1600 x 900 |
| 5 | `nifti_triview_graph.png` | Editor on a 4D BOLD file. Tri-view ON (sagittal + coronal + axial sharing one cyan crosshair) and Graph ON (time-series with volume marker mid-range). | 1600 x 900 |
| 6 | `validation.png` | Editor with the Validation pane on the right. Red errors plus amber warnings visible, severity chips in the toolbar with counts, red and amber badges on a few tree rows. | 1600 x 900 |
| 7 | `provenance.png` | A right-click "where did this come from?" tooltip on a converted file, or the project bundle being opened from a Recent menu. | 1200 x 700 |

> The wordmark used at the top of the README is already shipped in
> `bidsmgr/gui/assets/wordmark.png`. No capture needed for that one.

## Tips

- Hide the cursor unless it is the subject of the shot.
- Open a real dataset, not the synthetic test fixtures. The inspection
  table looks much more convincing with real sequence descriptions.
- For GIFs, keep them under 10 seconds and trim to the single action
  being demoed.
