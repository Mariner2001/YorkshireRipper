# Statistical foundations of Rossmo's formula

**Replication code for:**  
Úbeda-Flores, M. (2026). *Statistical foundations of Rossmo's formula with an application to the Yorkshire Ripper case.* Journal of Quantitative Criminology.

---

## Contents

| File | Description |
|------|-------------|
| `monte_carlo.py` | Reproduces Table 3 and Figure 1B (Monte Carlo convergence study) |
| `yorkshire_ripper.py` | Reproduces Table 2, Figure 1A, and Figure 2 (Yorkshire Ripper case study) |
| `requirements.txt` | Python dependencies |

---

## Requirements

Python ≥ 3.9 and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Usage

### Monte Carlo simulation (Table 3 + Figure 1B)

```bash
python monte_carlo.py
```

Runs 1,000 replications at each sample size n ∈ {5, 10, 20, 50, 100, 200, 500}.  
Expected runtime: ~90 seconds on a modern laptop.  
Output: printed table + `figure1B.pdf`.

### Yorkshire Ripper case study (Table 2 + Figures 1A, 2)

```bash
python yorkshire_ripper.py
```

Output: printed table + `figure1A.pdf` + `figure2.pdf`.

---

## Data

Crime-site coordinates (Table 1 of the paper) are hard-coded in `yorkshire_ripper.py`.  
Sources: Byford Report (Home Office, 1981) and Bilton (2003).  
All locations are in the public domain.

---

## License

MIT License. See `LICENSE`.
