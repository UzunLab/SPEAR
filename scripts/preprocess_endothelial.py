"""Preprocess endothelial scRNA/scATAC SingleCellExperiment RDS into AnnData.

This mirrors the embryonic preprocessing: barcode alignment, RNA MT filter,
and min-gene/min-cell filtering before writing combined QC h5ad files.
"""

from pathlib import Path
import os
import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import rpy2.robjects as ro
from rpy2.robjects import r

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
_LOG = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "endothelial" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "endothelial" / "processed"
WORK_DIR = PROJECT_ROOT / "work"

RNA_RDS = RAW_DIR / "sce_rna.rds"
ATAC_RDS = RAW_DIR / "sce_atac.rds"
RNA_REF_H5AD = RAW_DIR / "endo_rna_only.h5ad"

OUTPUT_RNA = PROCESSED_DIR / "combined_RNA_qc.h5ad"
OUTPUT_ATAC = PROCESSED_DIR / "combined_ATAC_qc.h5ad"

INTERMEDIATE_RNA = WORK_DIR / "sce_rna.h5ad"
INTERMEDIATE_ATAC = WORK_DIR / "sce_atac.h5ad"

# Mitochondrial filtering threshold (% of counts from mitochondrial genes)
# WARNING: This threshold is tuned for the specific endothelial dataset to yield ~5000 cells.
# For other datasets (especially different cell types, tissues, or sequencing platforms),
# this threshold may need adjustment. Consider examining the pct_counts_mt distribution
# before running preprocessing, and, if needed, override this default via a command-line
# argument or a configuration parameter in the calling workflow.
MT_THRESHOLD = 15.0


def _dgC_to_csc(r_mat) -> sp.csc_matrix:
    """Convert an R dgCMatrix (or coercible) into a SciPy csc_matrix."""
    ro.globalenv["mat"] = r_mat
    r("mat <- as(mat, 'dgCMatrix')")
    data = np.asarray(r("mat@x"), dtype=np.float32)
    indices = np.asarray(r("mat@i"), dtype=np.int32)
    indptr = np.asarray(r("mat@p"), dtype=np.int32)
    dims = tuple(r("mat@Dim"))
    return sp.csc_matrix((data, indices, indptr), shape=dims)


def _load_sce(path: Path) -> ad.AnnData:
    """Load a SingleCellExperiment RDS into AnnData (cells x features)."""
    r("library(SingleCellExperiment)")
    sce = r["readRDS"](str(path))
    ro.globalenv["sce"] = sce

    mat = _dgC_to_csc(r("assay(sce, 'X')"))
    var_names = list(r("rownames(sce)"))
    obs_names = list(r("colnames(sce)"))

    adata = ad.AnnData(X=mat.T)
    adata.obs_names = obs_names
    adata.var_names = var_names

    # Preserve basic metadata
    adata.obs["barcode"] = obs_names
    return adata


def _add_gene_symbols(adata_rna: ad.AnnData, reference: Path) -> None:
    """Attach gene symbols to RNA var using an existing h5ad reference."""
    ref = ad.read_h5ad(reference)
    mapping = pd.Series(ref.var["gene_ids"].values, index=ref.var_names)
    adata_rna.var["gene_ids"] = mapping.reindex(adata_rna.var_names).astype(str)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    adata_rna = _load_sce(RNA_RDS)
    adata_atac = _load_sce(ATAC_RDS)

    _add_gene_symbols(adata_rna, RNA_REF_H5AD)
    adata_atac.var["gene_ids"] = adata_atac.var_names

    # Store raw conversions for reproducibility/debugging.
    adata_rna.write_h5ad(INTERMEDIATE_RNA)
    adata_atac.write_h5ad(INTERMEDIATE_ATAC)

    # Align barcodes across modalities.
    common = adata_rna.obs_names.intersection(adata_atac.obs_names)
    adata_rna = adata_rna[common].copy()
    adata_atac = adata_atac[common].copy()

    # RNA QC: mitochondrial fraction filter (mirrors embryonic preprocessing).
    adata_rna.var["mt"] = adata_rna.var["gene_ids"].str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata_rna, qc_vars=["mt"], inplace=True)
    
    cells_before_mt_filter = adata_rna.n_obs
    adata_rna = adata_rna[adata_rna.obs.pct_counts_mt < MT_THRESHOLD].copy()
    cells_after_mt_filter = adata_rna.n_obs
    _LOG.info(
        "Mitochondrial QC filtering | threshold=%.1f%% | cells: %d -> %d (removed %d, %.1f%%)",
        MT_THRESHOLD,
        cells_before_mt_filter,
        cells_after_mt_filter,
        cells_before_mt_filter - cells_after_mt_filter,
        100.0 * (cells_before_mt_filter - cells_after_mt_filter) / cells_before_mt_filter if cells_before_mt_filter > 0 else 0.0
    )
    adata_rna.var_names_make_unique()

    # Simple filtering thresholds (min genes/cells).
    sc.pp.filter_cells(adata_rna, min_genes=200)
    sc.pp.filter_genes(adata_rna, min_cells=3)
    sc.pp.filter_cells(adata_atac, min_genes=200)
    sc.pp.filter_genes(adata_atac, min_cells=3)

    # Re-align after QC to keep paired cells only.
    common_post_qc = adata_rna.obs_names.intersection(adata_atac.obs_names)
    adata_rna = adata_rna[common_post_qc].copy()
    adata_atac = adata_atac[common_post_qc].copy()

    adata_rna.obs["sample"] = "endothelial"
    adata_atac.obs["sample"] = "endothelial"

    adata_rna.write_h5ad(OUTPUT_RNA)
    adata_atac.write_h5ad(OUTPUT_ATAC)

    print(
        f"Wrote RNA: {OUTPUT_RNA} (cells={adata_rna.n_obs}, genes={adata_rna.n_vars})"
    )
    print(
        f"Wrote ATAC: {OUTPUT_ATAC} (cells={adata_atac.n_obs}, peaks={adata_atac.n_vars})"
    )


if __name__ == "__main__":
    main()
