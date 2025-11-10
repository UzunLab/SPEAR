# Mouse Embryonic Stem Cell (mESC) Multi-omics Dataset

## Overview

- Single-cell paired RNA-seq and ATAC-seq profiling of mouse embryonic stem cell differentiation.
- Captures developmental time points spanning E7.5 through E8.75 with CRISPR perturbation controls.
- Organism & strain: C57BL/6Babr mice.

## Source References

- Primary study: _Decoding gene regulation in the mouse embryo using single-cell multi-omics_ ([bioRxiv, 2022](https://www.biorxiv.org/content/10.1101/2022.06.15.496239v2)).
- GEO accession: [GSE205117](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205117).

## Storage Layout

- Raw files (per-sample directories created by the download script):  
  `/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS/DS014_DOI496239_MOUSE_ESC_RAW_FILES`
- Processed AnnData objects and curated matrices:  
  `/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS/DS014_DOI496239_MOUSE_ESCDAYS7AND8`

## Sample Inventory

| Label            | GEO sample              | Modalities                                        |
| ---------------- | ----------------------- | ------------------------------------------------- |
| E7.5_rep1        | GSM6205416 / GSM6205427 | GEX (barcodes, features, matrix) + ATAC fragments |
| E7.5_rep2        | GSM6205417 / GSM6205428 | GEX + ATAC fragments                              |
| E7.75_rep1       | GSM6205418 / GSM6205429 | GEX + ATAC fragments                              |
| E8.0_rep1        | GSM6205419 / GSM6205430 | GEX + ATAC fragments                              |
| E8.0_rep2        | GSM6205420 / GSM6205431 | GEX + ATAC fragments                              |
| E8.5_CRISPR_T_KO | GSM6205421 / GSM6205432 | GEX + ATAC fragments                              |
| E8.5_CRISPR_T_WT | GSM6205422 / GSM6205433 | GEX + ATAC fragments                              |
| E8.5_rep1        | GSM6205423 / GSM6205434 | GEX + ATAC fragments                              |
| E8.5_rep2        | GSM6205424 / GSM6205435 | GEX + ATAC fragments                              |
| E8.75_rep1       | GSM6205425 / GSM6205436 | GEX + ATAC fragments                              |
| E8.75_rep2       | GSM6205426 / GSM6205437 | GEX + ATAC fragments                              |

## Downloading Raw Data

1. Ensure you are on a Slurm login node with outbound FTP access.
2. Submit the Slurm batch script that orchestrates per-sample downloads:

   ```bash
   sbatch jobs/download_mesc_raw_data.sbatch
   ```

3. Monitor progress via the script log files (`/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/download_raw_mesc_files.{log,err}`).
4. Verify that each sample directory contains the expected `*.tsv.gz` or `*.mtx.gz` files before launching downstream preprocessing.

## Notes

- The script skips files that already exist, allowing safe reruns if a transfer fails.
- Update partition, memory, or output paths in the Slurm header if running on a different cluster.
- Processed objects in the curated directory should be refreshed after re-downloading raw data to avoid mismatches.
