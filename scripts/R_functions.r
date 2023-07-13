library(edgeR)
library(limma)

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
    }
    
    if (design == "none")
        keep <- filterByExpr(y)
    else
        keep <- filterByExpr(y, design=design)
    
    y <- y[keep,,keep.lib.sizes=FALSE]
    write.csv(y$counts, outpath, row.names = TRUE)
}