args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Expected author RDS path and output TSV.GZ path")
expression <- readRDS(args[[1]])
if (!is.data.frame(expression) || names(expression)[[1]] != "symbol" ||
    anyDuplicated(names(expression)) || any(!vapply(expression[-1], is.numeric, logical(1)))) {
    stop("Unexpected author expression schema")
}
out <- gzfile(args[[2]], open = "wt")
write.table(expression, out, sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
close(out)
