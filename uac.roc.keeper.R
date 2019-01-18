
#############################################

#------------------
# PREDICTION TYPES
#------------------
# Source: https://www.joyofdata.de/blog/illustrated-guide-to-roc-and-auc/

plot_pred_type_distribution <- function(plist, label, cutoff) {
  # plist: list values of predictions
  # label: list of label values
  # thre
  pred_type <- rep(NA, length(plist))
  pred_type <- ifelse(plist >= cutoff & label == 1, "TP",
                      ifelse(plist >= cutoff & label == 0, "FP",
                             ifelse(plist < cutoff & label == 1, "FN",
                                    ifelse(plist < cutoff & label == 0, "TN", '??'))))

  ggplot(data=as.data.frame(cbind(plist,label)), aes(x=label, y=plist))
  + geom_violin(fill=rgb(1,1,1,alpha=0.5), color=NA)
  + geom_jitter(aes(color=pred_type), alpha=0.5)
  + geom_hline(yintercept=cutoff, color="red", alpha=0.6)
  + scale_color_discrete(name = "type")
  + labs(title=sprintf("Threshold at %.2f", cutoff)
         ,y='prediction')
}

threshold <- 0.5

ggnnls <- ggplotly(
  plot_pred_type_distribution(pred3[,'nnls'],y.test, threshold)
);ggnnls

ggauc <- ggplotly(
  plot_pred_type_distribution(pred3[,'auc'],y.test, threshold)
);ggauc

ggnnloglik <- ggplotly(
  gplot_pred_type_distribution(pred3[,'nnloglik'],y.test, threshold)
);ggnnloglik

# https://github.com/joyofdata/joyofdata-articles/blob/master/roc-auc/calculate_roc.R
calculate_roc <- function(plist, label, cutoff, cost_of_fp, cost_of_fn) {

  n <- length(plist)

  tpr <- function(plist, cutoff) {
    sum(plist >= cutoff & label == 1) / sum(label == 1)
  }

  fpr <- function(plist, cutoff) {
    sum(plist >= cutoff & label== 0) / sum(label == 0)
  }

  cost <- function(plist, cutoff, cost_of_fp, cost_of_fn) {
    sum(plist >= cutoff & label == 0) * cost_of_fp
    + sum(plist >= cutoff & label == 1) * cost_of_fn
  }

  roc <- data.frame(cutoff = seq(0,1,length.out=n), tpr=NA, fpr=NA)
  roc$tpr <- sapply(roc$cutoff, function(th) tpr(plist, th))
  roc$fpr <- sapply(roc$cutoff, function(th) fpr(plist, th))
  roc$cost <- sapply(roc$cutoff, function(th) cost(plist, th, cost_of_fp, cost_of_fn))

  return(roc)
}

roc <- calculate_roc(pred3[,'nnls'], y.test, threshold, 1, 2)

# https://github.com/joyofdata/joyofdata-articles/blob/master/roc-auc/plot_roc.R
plot_roc <- function(roc, threshold, cost_of_fp, cost_of_fn) {

  if (!require('grid')) install.packages('grid'); library(grid)
  if (!require('gridExtra')) install.packages('gridExtra'); library(gridExtra)

  norm_vec <- function(v) (v - min(v))/diff(range(v))

  idx_threshold = which.min(abs(roc$threshold-threshold))

  col_ramp <- colorRampPalette(c("green","orange","red","black"))(100)
  col_by_cost <- col_ramp[ceiling(norm_vec(roc$cost)*99)+1]
  p_roc <- ggplot(roc, aes(fpr,tpr)) +
    geom_line(color=rgb(0,0,1,alpha=0.3)) +
    geom_point(color=col_by_cost, size=4, alpha=0.5) +
    coord_fixed() +
    geom_line(aes(threshold,threshold), color=rgb(0,0,1,alpha=0.5)) +
    labs(title = sprintf("ROC")) + xlab("FPR") + ylab("TPR") +
    geom_hline(yintercept=roc[idx_threshold,"tpr"], alpha=0.5, linetype="dashed") +
    geom_vline(xintercept=roc[idx_threshold,"fpr"], alpha=0.5, linetype="dashed")

  p_cost <- ggplot(roc, aes(threshold, cost)) +
    geom_line(color=rgb(0,0,1,alpha=0.3)) +
    geom_point(color=col_by_cost, size=4, alpha=0.5) +
    labs(title = sprintf("cost function")) +
    geom_vline(xintercept=threshold, alpha=0.5, linetype="dashed")

  sub_title <- sprintf("threshold at %.2f - cost of FP = %d, cost of FN = %d", threshold, cost_of_fp, cost_of_fn)

  grid.arrange(p_roc, p_cost, ncol=2, sub=textGrob(sub_title, gp=gpar(cex=1), just="bottom"))
}

plot_roc(roc, threshold, 1, 2)

#----------------
# AUC of Method
#----------------
if (!require('pROC')) install.packages('pROC'); library(pROC)
auc.method <- noquote(cbind(
  c('nnls','auc','nnloglik')
  ,c(
    auc(y.test, pred3[,'nnls'])[1]
    ,auc(y.test, pred3[,'auc'])[1]
    ,auc(y.test, pred3[,'nnloglik'])[1]
  )));colnames(auc.method) <- c('method','area under the curve') ;auc.method

#########################################