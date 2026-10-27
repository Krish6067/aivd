# Datasets

This directory is reserved for anti-spoofing training and evaluation datasets.

## ⚠ Do Not Commit Datasets to Version Control

Datasets are excluded by `.gitignore`. Store them locally only.

---

## ASVspoof 2019 (used to train AASIST)

| Property | Value |
|---|---|
| **Name** | ASVspoof 2019 |
| **Partition used** | Logical Access (LA) |
| **License** | Research use only |
| **Access** | Application required via official website |
| **Website** | https://www.asvspoof.org/ |
| **Size** | ~10 GB (LA partition) |

### How to obtain

1. Visit https://www.asvspoof.org/
2. Register and complete the data access agreement.
3. Download the LA partition.
4. Place under `datasets/ASVspoof2019/LA/`.

### Directory layout (after download)

```
datasets/
└── ASVspoof2019/
    └── LA/
        ├── ASVspoof2019_LA_train/
        ├── ASVspoof2019_LA_dev/
        ├── ASVspoof2019_LA_eval/
        └── ASVspoof2019_LA_cm_protocols/
```

---

## ASVspoof 2021 (optional for evaluation)

| Property | Value |
|---|---|
| **Website** | https://www.asvspoof.org/ |
| **License** | Research use only |

---

## Note on Evaluation

Using ASVspoof data for evaluating a deployed production system requires
careful consideration of the dataset licence and distribution shift.
The VOXSHIELD backend is not a substitute for a properly evaluated,
audited anti-spoofing system.
