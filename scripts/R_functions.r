library(edgeR)
library(limma)
library(dplyr)
library(arrow)

save_table <- function(table, outfile) {
    suppressPackageStartupMessages(require("feather"))
    outfile_ext <- strsplit(basename(outfile), split="\\.")[[1]]
    outfile_ext <- outfile_ext[length(outfile_ext)]
    if (outfile_ext == "feather") {
        table = as.data.frame(table)
        table$Row = row.names(table)
        write_feather(table, outfile)  
    }
    else {
        write.csv(table, outfile, row.names = TRUE)
    }
}

#' Use edgeR's filterByExpr to remove low count genes from count matrix
#' 
#' @param inpath String specifying path to csv file of gene expression data
#' @param outpath String speciffying path to store filtered csv file
#' @param design Design matrix
edgeR_filterByExpression <- function(inpath, outpath, design) {
    
    x <- read.csv(inpath, row.names=1)
    x <- data.matrix(x, rownames.force=TRUE)
    y <- DGEList(counts=x)
    
    if (design == "paired") {
        if (ncol(x)%%2 != 0) {stop("Paired-design matrix must have even number of columns")}
        N <- ncol(x)/2
        patient <- factor(c(seq(N),seq(N)))
        condition <- factor(c(rep("N",N),rep("T",N))) # normal vs tumor (control vs treatment)
        design <- model.matrix(~patient+condition)
    } else if (design=="unpaired") {
        if (ncol(x)%%2 != 0) {stop("Design matrix must have even number of columns")}
        N <- ncol(x)/2
        condition <- factor(c(rep("N",N),rep("T",N))) # normal vs tumor (control vs treatment)
        design <- model.matrix(~condition)
    }
    
    if (design == "none")
        keep <- filterByExpr(y)
    else
        keep <- filterByExpr(y, design=design)
    
    y <- y[keep,,keep.lib.sizes=FALSE]
    write.csv(y$counts, outpath, row.names = TRUE)
}

#' Run edgeR
#'
#' @param x: dataframe of counts
#' @param design: design matrix, if "paired" constructs design matrix from data assuming x is of the form: k control cols followed by k treatment cols
#' @param overwrite: logical, whether to overwrite existing results table if present
#' @param filter_expr: logical, whether to remove low counts using edgeR's filterByExpr function
#' @param top_tags: int or "Inf", store the results of the most significant genes only
#' @param lfc: float, logFC threshold when testing for DE
#' @param cols_to_keep: list of output table columns to save
run_edgeR <- function(x, outfile, design, overwrite=FALSE, filter_expr=FALSE, top_tags = "Inf", verbose=FALSE,
                      lfc=0, cols_to_keep="all", test="qlf", meta_only=FALSE, check_gof=FALSE, N_control=0, N_treat=0) {

    suppressPackageStartupMessages(require("edgeR"))
    suppressPackageStartupMessages(require("limma"))
    
    # Check if files already exist
    if (!overwrite && file.exists(outfile)) {
        print("Existing table not overwritten")
        return()
    }

    if (design == "paired") {
        if (ncol(x)%%2 != 0) {stop("Paired-design matrix must have even number of columns")}
        N <- ncol(x)/2
        patient <- factor(c(seq(N),seq(N)))
        condition <- factor(c(rep("N",N),rep("T",N))) # normal vs tumor (control vs treatment)
        design <- model.matrix(~patient+condition)
    } else if (design=="unpaired") {
        if (ncol(x)%%2 != 0) {stop("Design matrix must have even number of columns")}
        N <- ncol(x)/2
        condition <- factor(c(rep("N",N),rep("T",N)))
        design <- model.matrix(~condition)
    } else if (design=="unpaired_asymmetric") {
        if (N_control == 0) {stop("Design matrix has no control columns")}
        if (N_treat == 0) {stop("Design matrix has no treatment/condition columns")}
        condition <- factor(c(rep("N",N_control),rep("T",N_treat)))
        design <- model.matrix(~condition)
    } else if (grepl("\\.csv$", design, ignore.case = TRUE)) {
        
        print("Constructing design matrix from df")
        covariate_df = read.csv(design)
        if (!("Condition" %in% colnames(covariate_df))) {stop("Error: 'Condition' column not found in dataframe")}
        
        covariate_df <- covariate_df %>%
          mutate_if(is.character, as.factor)
        
        other_vars <- setdiff(names(covariate_df), c("Condition", "X", "Sample"))
        print("Warning: hard-coded col names in design matrix")
        formula <- as.formula(paste("~", paste(c(other_vars, "Condition"), collapse = " + ")))
        print(paste("Formula:",formula))
        design <- model.matrix(formula, data = covariate_df)
    }
    
    if (verbose) {
        rank = qr(design)$rank
        print(paste("Rank:",rank, "Cols:", ncol(design)))
        print(design)
        }
    
    y <- DGEList(counts=x)

    print(length(rownames(design)))
    print(length(colnames(y)))
    rownames(design) <- colnames(y)

    if (filter_expr) {
        keep <- filterByExpr(y, design=design)
        y <- y[keep,,keep.lib.sizes=FALSE]
    }

    y <- calcNormFactors(y)
    y <- estimateDisp(y, design, robust=TRUE)
    
    if (meta_only)
        return(y)

    if (test=="lrt") fit <- glmFit(y,design)
    else fit <- glmQLFit(y,design)       

    # Goodness-of-fit
    if (check_gof) {
        res.gof <- gof(fit, plot=FALSE)
        file_name <- basename(outfile)
        new_file_name <- paste0("gof.", file_name)
        new_file_path <- file.path(dirname(outfile), new_file_name)
        save_table(res.gof$gof.pvalues, new_file_path)
        print(paste("Saved gof in", new_file_path))
    }

    if (lfc>0) result <- glmTreat(fit, lfc=lfc)
    else if (test=="lrt") result <- glmLRT(fit)
    else result <- glmQLFTest(fit) # omit coef (edgeR user's guide p. 39)

    table = topTags(result, n = top_tags) #adjust.method="BH"
    
    if (any(cols_to_keep != "all")) {
        if (typeof(cols_to_keep)=="list") cols_to_keep = unlist(cols_to_keep)
        table <- table[ , cols_to_keep]
    }
    save_table(table, outfile)
}


# DESeq2
run_deseq2 <- function(x, outfile, design="paired", overwrite=FALSE, print_summary=FALSE, cols_to_keep="all", size_factors_only=FALSE, lfc=0) {
    
    if (!overwrite && file.exists(outfile)) {
        print("Existing table not overwritten")
        return()
    }
    
    suppressPackageStartupMessages(require("DESeq2"))

    if (design == "paired") {
        if (ncol(x)%%2 != 0) {stop("Paired-design matrix must have even number of columns")}
        N <- ncol(x)/2
        patients <- factor(c(seq(N),seq(N)))
        condition <- factor(c(rep("N",N),rep("T",N))) # normal vs tumor (control vs treatment)
        array = array(factor(c(patients,condition)),dim=c(length(patients),2))
        coldata <- data.frame(array, row.names = colnames(x))
        colnames(coldata) <- c("patient","condition")
        
        dds <- DESeqDataSetFromMatrix(countData = x,
                                  colData = coldata,
                                  design = ~ patient + condition)
    }
    else if (design == "unpaired") {
        if (ncol(x)%%2 != 0) {stop("Design matrix must have even number of columns")}
        N <- ncol(x)/2
        condition <- factor(c(rep("N",N),rep("T",N))) # normal vs tumor (control vs treatment)
        array = array(factor(condition),dim=c(length(condition)))
        coldata <- data.frame(array, row.names = colnames(x))
        colnames(coldata) <- c("condition")
        dds <- DESeqDataSetFromMatrix(countData = x,
                                  colData = coldata,
                                  design = ~ condition)
    }
    else {stop("Test not implemented")}
    

    
    if (size_factors_only)
        return(sizeFactors(estimateSizeFactors(dds)))
    
    dds <- DESeq(dds)
    res <- results(dds, name="condition_T_vs_N", lfcThreshold=lfc, altHypothesis="greaterAbs", test="Wald")
    
    if (print_summary) {print(summary(res))}
    
    names(res)[names(res) == "padj"] <- "FDR"
    names(res)[names(res) == "log2FoldChange"] <- "logFC"
    #names(res)[names(res) == "pvalue"] <- "PValue"
    names(res)[names(res) == "baseMean"] <- "logCPM" # !!!! for consistent naming with edgeR
   
    if (any(cols_to_keep != "all")) {
        if (typeof(cols_to_keep)=="list") cols_to_keep = unlist(cols_to_keep)
        res <- res[ , cols_to_keep]
    }
    
    save_table(res, outfile) 
}

jackknife_paired <- function(x, outname, path, overwrite=FALSE, include_full=TRUE, DEA="edgerqlf", skip_cols="skip_none", ...) {
# run jackknife on paired-design experiment data

    if (!overwrite) {
        mergedFile = paste(path,"/",outname,"_jacked_merged.csv",sep="")
        if (file.exists(mergedFile)) {
            print(paste(mergedFile,"found (skipping)"))
            return(1)
        }
    }

    replicates = length(x)/2
    controls = colnames(x)[1:replicates]
    
    for (control in controls) {
        if (control %in% skip_cols) {
            #print(paste("skipping",control))
            next
            }
        
        # Define file paths and check if files already exist
        tableFile = paste(path,"/",outname,"_",control,"_table.csv",sep="")
        if (!overwrite && file.exists(tableFile)) next  
        
        # Remove currently iterated patient
        ind <- which(colnames(x)==control)
        ind <- c(ind, ind+replicates)
        xx <- x[-ind]

        # Do DE analysis
        if (DEA == "edgerqlf") {
            run_edgeR(xx, tableFile, design="paired", overwrite=overwrite, ...) 
        } else if (DEA == "DESeq2") {
            run_deseq2(xx, tableFile, design="paired", overwrite=overwrite, ...)
        }
        else {stop(paste(dea,"not implemented"))}
        
    }
    
    # Do also DEG analysis for full dataset (no patient removed)
    if (include_full) {
        tableFile = paste(path,"/",outname,"_0_table.csv",sep="")
        if (overwrite || !file.exists(tableFile)) {
            if (DEA == "edgerqlf") {
                run_edgeR(x, tableFile, design="paired", overwrite=overwrite, cols_to_keep=c("logFC","logCPM","FDR")) 
            }
            else if (DEA == "DESeq2") {
                run_deseq2(x, tableFile, design="paired", overwrite=overwrite, cols_to_keep=c("logFC","logCPM","FDR"))
            }
        }
    }
    return(0)
}


# Molecular degree of perturbation
# https://bioconductor.org/packages/devel/bioc/vignettes/mdp/inst/doc/my-vignette.html
# https://rdrr.io/bioc/mdp/src/R/mdp.R
mdp_analysis <- function(df, pheno, control_lab, plot=FALSE, save_tables=FALSE, directory="", file_name="", all_genes=FALSE, std = 2, fraction_genes = 0.25, measure="mean") {
    # df: count dataframe with gene names a row names, sample names as col names; index col must be named "Symbol"
    # pheno: df with at least two cols: "Sample" = df col names, "Class" = class labels
    # control_lab = control label, must match a class label
    
    suppressPackageStartupMessages(library(mdp))
    
    mdp.results <- mdp(data=df, pdata=pheno, control_lab=control_lab, print=FALSE, save_tables=save_tables, directory=directory, file_name=file_name, std=std, fraction_genes=fraction_genes,measure=measure)
    
    sample_scores_list <- mdp.results$sample_scores
    if (all_genes) sample_scores <- sample_scores_list[["allgenes"]]
    else sample_scores <- sample_scores_list[["perturbedgenes"]]
    
    if (plot) sample_plot(sample_scores, filename = file_name,
                        directory = directory, title = "", print = TRUE,
                        display = FALSE, control_lab)

#     print(sample_scores)
#     outliers <- sample_scores[sample_scores$outlier != 0, ] 
#     if (plot) print(head(outliers))
#     outliers <- rownames(outliers)
    return(sample_scores)
}


# Robust PCA
pcahubert <- function(df, k=0, plot=FALSE) {
    suppressPackageStartupMessages(require(rrcov))
    tdf <- t(df)
    pca <- PcaHubert(tdf,k=k)
    outliers <- which(pca@flag=='FALSE')   
    
    if (plot) {
            plot(pca)                    # distance plot
            #pca2 <- PcaHubert(tdf, k=2)  
            #plot(pca2)                   # PCA diagnostic plot (or outlier map)
    
            ## Use the standard plots available for prcomp and princomp
            #screeplot(pca)    
            #biplot(pca)    
    
    }
    return(outliers)
}


# Customized enrichKEGG from clusterProfiler to use local KEGG data
enrichKEGG_custom <- function(gene,
                       organism          = "hsa",
                       keyType           = "kegg",
                       pvalueCutoff      = 0.05,
                       pAdjustMethod     = "BH",
                       universe,
                       minGSSize         = 10,
                       maxGSSize         = 500,
                       qvalueCutoff      = 0.2,
                       use_internal_data = FALSE,
                       internal_data_path = "") {

    suppressPackageStartupMessages(require(clusterProfiler))
    
    species <- organismMapper(organism)
    
    if (use_internal_data) {
        
        # Original code outdated since KEGG.db has been removed from Bioconductor
        #KEGG_DATA <- get_data_from_KEGG_db(species)
        
        # Custom alternative: download KEGG and save it locally
        if (!file.exists(internal_data_path)) {
            KEGG_DATA <- prepare_KEGG(species, "KEGG", keyType)
            save(KEGG_DATA, file=internal_data_path)
        }
        else {
            print(paste("Loading",internal_data_path))
            load(internal_data_path)
        }
        
    } else {
        KEGG_DATA <- prepare_KEGG(species, "KEGG", keyType)
    }
    
    res <- enricher_internal(gene,
                             pvalueCutoff  = pvalueCutoff,
                             pAdjustMethod = pAdjustMethod,
                             universe      = universe,
                             minGSSize     = minGSSize,
                             maxGSSize     = maxGSSize,
                             qvalueCutoff  = qvalueCutoff,
                             USER_DATA = KEGG_DATA)
    if (is.null(res))
        return(res)

    res@ontology <- "KEGG"
    res@organism <- species
    res@keytype <- keyType
    return(res)
}


clusterORA <- function(degs, universe, outname, go_ont="BP", prefix="clusterORA", pAdjustMethod="BH", minGSSize = 15, maxGSSize = 500,
                       overwrite=FALSE, use_internal_data=FALSE, internal_data_path = "") {
    
    outfile_go <- paste0(prefix,".GO_Biological_Process_2021",outname,".feather")
    outfile_kegg <- paste0(prefix,".KEGG_2021_Human",outname,".feather")
    
    # Check if files already exist
    if (!overwrite && file.exists(outfile_go) && file.exists(outfile_kegg)) {
        print("Existing clusterORA tables not overwritten")
        return()
    }
    suppressPackageStartupMessages(require("clusterProfiler"))
    suppressPackageStartupMessages(require("org.Hs.eg.db"))
    suppressPackageStartupMessages(require("AnnotationDbi"))
    suppressPackageStartupMessages(require("stats4"))
    suppressPackageStartupMessages(require("BiocGenerics"))
    
    # Customized enrichKEGG function to use local KEGG data
    environment(enrichKEGG_custom) <- asNamespace('clusterProfiler')
    assignInNamespace("enrichKEGG", enrichKEGG_custom, ns = "clusterProfiler")
    kegg <- enrichKEGG_custom(gene=degs, 
                       universe=names(universe), 
                       minGSSize = minGSSize, maxGSSize = maxGSSize,
                       organism="hsa", keyType = "ncbi-geneid", pvalueCutoff = 1, qvalueCutoff=1, pAdjustMethod=pAdjustMethod, use_internal_data=use_internal_data, internal_data_path = internal_data_path)
    
    # Regular enrichGO
    go <- enrichGO(gene = degs,
                      universe = names(universe),
                      OrgDb = "org.Hs.eg.db", 
                      keyType = 'ENTREZID',
                      readable = T,
                      ont = go_ont,
                      minGSSize = minGSSize, maxGSSize = maxGSSize,
                      pvalueCutoff = 1, qvalueCutoff = 1, pAdjustMethod=pAdjustMethod)
    
    if (go_ont != "BP") {
        print(paste("Warning:", go_ont, "go_ont is not BP but outfile name hardcoded as BP"))
    }
    
    go = as.data.frame(go)
    kegg = as.data.frame(kegg)
    
    names(go)[names(go) == "Description"] <- "Term"
    names(kegg)[names(kegg) == "Description"] <- "Term"
    names(go)[names(go) == "p.adjust"] <- "FDR"
    names(kegg)[names(kegg) == "p.adjust"] <- "FDR"
    
    go = subset(go, select = -ID)
    kegg = subset(kegg, select = -ID)  
    
    print(paste("saving in",outfile_go))
    print(paste("saving in",outfile_kegg))
    save_table(go, outfile_go)
    save_table(kegg, outfile_kegg)

}
