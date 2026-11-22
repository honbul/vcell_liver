#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(affy))

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 4L) stop("Usage: prepare_phh.R CEL_ROOT OUT_DIR ARRAY_INDEX PROBE_MAP")
root <- args[[1L]]
out <- args[[2L]]
shard <- as.integer(args[[3L]])
mapping <- read.delim(args[[4L]], colClasses="character")
stopifnot(!anyDuplicated(mapping$probe))
folders <- sort(list.dirs(root, recursive=FALSE))
if (is.na(shard) || shard < 0L || shard >= 8L) stop("Invalid array index")
folders <- folders[seq_along(folders) %% 8L == shard]
if (!dir.exists(out)) dir.create(out, recursive=TRUE)

process <- function(folder) {
  name <- basename(folder)
  meta <- read.delim(file.path(folder, "Attribute.tsv"), colClasses="character",
                     check.names=FALSE, fileEncoding="latin1", quote="")
  meta <- meta[which(meta$SPECIES == "Human" & meta$TEST_TYPE == "in vitro" &
                     meta$SACRI_PERIOD == "24 hr"), ]
  if (!nrow(meta)) return(data.frame(folder=name, status="no_24h", samples=0L, contrasts=0L))
  if (!all(meta$ARR_DESIGN %in% "HG-U133_Plus_2")) {
    return(data.frame(folder=name, status=paste0("alternate_array:", paste(unique(meta$ARR_DESIGN), collapse=",")),
                      samples=nrow(meta), contrasts=0L))
  }
  stopifnot(!anyDuplicated(meta$BARCODE))
  files <- file.path(folder, "celfiles", paste0(meta$BARCODE, ".CEL"))
  if (!all(file.exists(files))) stop("Missing pinned CEL under ", name)
  # RMA is local to an experiment and uses no arrays from held-out chemicals.
  by_experiment <- split(seq_len(nrow(meta)), meta$EXP_ID)
  results <- list()
  for (ii in by_experiment) {
    block <- meta[ii, ]
    controls <- which(block$DOSE_LEVEL == "Control")
    treated_rows <- which(!is.na(block$DOSE_LEVEL) & block$DOSE_LEVEL != "Control")
    groups <- split(treated_rows, block$DOSE_LEVEL[treated_rows])
    if (length(controls) < 2L || !length(groups)) next
    arrays <- ReadAffy(filenames=files[ii])
    expr <- exprs(rma(arrays, verbose=FALSE))
    probes <- rownames(expr)
    symbols <- mapping$symbol[match(probes, mapping$probe)]
    valid <- !is.na(symbols)
    vehicle <- rowMeans(expr[, controls, drop=FALSE])
    for (treated in groups) {
      if (length(treated) < 2L) next
      level <- unique(block$DOSE_LEVEL[treated])
      dose <- unique(as.numeric(block$DOSE[treated]))
      if (length(dose) != 1L || !is.finite(dose) || dose <= 0) stop("Dose mismatch in ", name)
      contrasts <- rowMeans(expr[, treated, drop=FALSE]) - vehicle
      delta <- tapply(contrasts[valid], symbols[valid], median)
      results[[length(results) + 1L]] <- list(
        meta=data.frame(compound=block$COMPOUND_NAME[[1L]], experiment=block$EXP_ID[[1L]],
                        dose_uM=dose, dose_level=level, treated=length(treated),
                        controls=length(controls), dna_percent=mean(as.numeric(block[["DNA(%)"]][treated]), na.rm=TRUE),
                        ldh_percent=mean(as.numeric(block[["LDH(%)"]][treated]), na.rm=TRUE),
                        treated_barcodes=paste(block$BARCODE[treated], collapse=","),
                        control_barcodes=paste(block$BARCODE[controls], collapse=",")),
        delta=delta)
    }
  }
  if (length(results)) {
    dest <- file.path(out, paste0(name, ".rds"))
    if (file.exists(dest)) stop("Destination exists ", dest)
    saveRDS(results, dest)
  }
  data.frame(folder=name, status=ifelse(length(results), "ok", "no_replicates"),
             samples=nrow(meta), contrasts=length(results))
}

inventory <- lapply(folders, process)
write.table(do.call(rbind, inventory), file.path(out, paste0("shard-", shard, ".tsv")),
            sep="\t", row.names=FALSE, quote=FALSE)
