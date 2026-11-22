#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(hgu133plus2.db))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1L) stop("Usage: phh_probe_map.R OUTPUT_TSV")
probes <- sort(keys(hgu133plus2.db, keytype="PROBEID"))
symbols <- mapIds(hgu133plus2.db, keys=probes, keytype="PROBEID",
                  column="SYMBOL", multiVals="asNA")
write.table(data.frame(probe=probes, symbol=unname(symbols)), args[[1L]],
            sep="\t", row.names=FALSE, quote=FALSE)
