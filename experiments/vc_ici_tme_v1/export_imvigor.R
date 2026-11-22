# Read the author-deposited CountDataSet without installing legacy DESeq.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
    stop("usage: Rscript export_imvigor.R cds.RData counts.tsv clinical.tsv")
}
scratch <- new.env(parent = baseenv())
loaded <- suppressWarnings(load(args[[1]], envir = scratch))
if (!identical(loaded, "cds") || !identical(class(scratch$cds)[[1]], "CountDataSet")) {
    stop("Unexpected IMvigor210CoreBiologies object")
}
assays <- attr(scratch$cds, "assayData")
counts <- get("counts", envir = assays, inherits = FALSE)
clinical <- attr(attr(scratch$cds, "phenoData"), "data")
if (!is.matrix(counts) || !is.numeric(counts) || any(counts < 0) || any(!is.finite(counts)) ||
    !identical(colnames(counts), rownames(clinical))) {
    stop("Invalid or misaligned deposited IMvigor210 expression")
}
fields <- c("ANONPT_ID", "Best Confirmed Overall Response")
if (!all(fields %in% names(clinical))) {
    stop("IMvigor210 response metadata is incomplete")
}
write.table(counts, args[[2]], quote = FALSE, sep = "\t", col.names = NA)
write.table(clinical[, fields, drop = FALSE], args[[3]], quote = FALSE, sep = "\t", col.names = NA)
