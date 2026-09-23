# Synthetic Data Generation Platform

Frontend for the insurance CTGAN project. Eight pages from the layout mock, live-wired to `insurance.csv`.

## Run

```bash
python -m http.server 8000
```

Open http://localhost:8000

Do not open `index.html` as a file URL — the browser must fetch `insurance.csv`.

## Pages

Dashboard, Dataset overview, Data analysis, CTGAN training, Synthetic data, Comparison, Visualizations, Documentation.

**Train** then **Generate** to refresh synthetic rows. **Download CSV** exports them.

The notebook (`ctgan_project.ipynb`) trains real CTGAN with PyTorch. This UI uses a fast in-browser generator that preserves smoker/sex/charges structure so the app works without GPU.
