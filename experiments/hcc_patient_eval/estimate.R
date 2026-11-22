#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(estimate))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3L) stop("Usage: estimate.R INPUT_TSV FILTERED_GCT SCORE_GCT")
filterCommonGenes(args[[1L]], args[[2L]], id="GeneSymbol")
# The original ESTIMATE immune/stromal signatures use the Illumina branch for RNA-seq.
estimateScore(args[[2L]], args[[3L]], platform="illumina")
