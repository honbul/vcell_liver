#!/usr/bin/env Rscript
# Score gene sets with the published singscore implementation.
# args: <expr.tsv: genes x samples, first column gene symbol> <gmt> <out.tsv: samples x sets>
suppressPackageStartupMessages(library(singscore))
args <- commandArgs(trailingOnly = TRUE)
expr <- read.delim(args[1], row.names = 1, check.names = FALSE)
lines <- strsplit(readLines(args[2]), "\t")
ranked <- rankGenes(as.matrix(expr), tiesMethod = "min")
out <- matrix(NA_real_, nrow = ncol(expr), ncol = length(lines),
              dimnames = list(colnames(expr), vapply(lines, `[`, "", 1)))
for (i in seq_along(lines)) {
  genes <- lines[[i]][-(1:2)]
  stopifnot(all(genes %in% rownames(expr)))
  sc <- simpleScore(ranked, upSet = genes, centerScore = TRUE)
  stopifnot(identical(rownames(sc), colnames(expr)))
  out[, i] <- sc$TotalScore
}
write.table(data.frame(model_id = rownames(out), out, check.names = FALSE), args[3],
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("singscore", as.character(packageVersion("singscore")), R.version.string, "\n")
