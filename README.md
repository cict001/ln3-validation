# ln(3) Validation Code

Replication code for the ln(3) coordination threshold papers.

**Author:** Jian Ji, CICT | jijian1@cictci.com | ORCID: 0009-0002-3735-3697  
**Preprint:** arXiv:2603.15521 | **License:** MIT

---

## ▶ One-click reproduction (no installation needed)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cict001/ln3-validation/blob/main/reproduce_main_results.ipynb)

Click the badge above to open in Google Colab and run all key results in your browser.

**What you can verify without downloading any data:**
- Core theorem: ln(3) is the unique solution to 2/(e^x − 1) = 1
- ρ_c/ρ_j = 0.5093 (analytical derivation)
- Traffic MAPE = 1.9% across 4 datasets (using published jam densities)
- Chengdu aggregate statistics internal consistency

**What requires data download:**
- UTD19 Constance full analysis (~500MB, free from utd19.ethz.ch)
- highD/NGSIM fitting (registration required)

---

## Repository Structure

    reproduce_main_results.ipynb   ← START HERE (Colab notebook)
    analysis/                      ← Core analysis scripts
    figures/                       ← Figure generation scripts
    supplement/                    ← Supplementary analysis

---

## Datasets

| Dataset         | Source                    | DOI / URL                   |
|-----------------|---------------------------|-----------------------------|
| highD (Germany) | levelxdata.com            | 10.17632/p6nkp3kkmk.2      |
| NGSIM I-80 (US) | data.transportation.gov   | 10.21949/1504477            |
| UTD19 Constance | utd19.ethz.ch             | 10.1038/s41597-019-0001-9   |
| Rope3D          | thudair.baai.ac.cn/rope   | CVPR 2022                   |
| DAIR-V2X        | thudair.baai.ac.cn/index  | CVPR 2022                   |

**Chengdu V2X:** CICT/Chengdu operational data, not publicly available.  
Aggregate statistics reported in papers. Contact jijian1@cictci.com for academic verification.

---

## Local Installation

    pip install numpy matplotlib scipy pandas
    python analysis/analyze_UTD19_final.py --data_dir /path/to/UTD19
    python figures/generate_figures_Nature.py

---

## Pre-registration Records

| Prediction              | Zenodo DOI                    |
|------------------------|-------------------------------|
| LFP critical SOC 63.38% | 10.5281/zenodo.19212251      |
| Theory preprint         | 10.5281/zenodo.19101912       |

---

## Citation

    @misc{Ji2026ln3,
      author = {Ji, Jian},
      title  = {A topological threshold for bidirectional coordination
                in one-dimensional Poisson proximity networks},
      year   = {2026},
      note   = {arXiv:2603.15521. Under review, Phys. Rev. Lett., LQ20428}
    }
