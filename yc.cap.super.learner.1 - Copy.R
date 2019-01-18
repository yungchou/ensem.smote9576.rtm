if (!require('SuperLearner')) install.packages('SuperLearner'); library(SuperLearner)
if (!require('ranger' )) install.packages('ranger' ); library(ranger )
if (!require('xgboost')) install.packages('xgboost'); library(xgboost)
if (!require('caret'  )) install.packages('caret'  ); library(caret  )

if(TRUE){ # skip the following

if (!require('mlbench')) install.packages('mlbench'); library(mlbench)
if (!require('rpart')) install.packages('rpart'); library(rpart)
if (!require('rUnit')) install.packages('rUnit'); library(rUnit)

if (!require('h2o')) install.packages('h2o'); library(h2o)
if (!require('gbm')) install.packages('gbm'); library(gbm)

  install.packages("devtools");library(devtools
);install_github("AppliedDataSciencePartners/xgboostExplainer")

if (!require('MASS')) install.packages('MASS'); library(MASS)
if (!require('cvAUC')) install.packages('cvAUC'); library(cvAUC)
if (!require('ggplot2')) install.packages('ggplot2'); library(ggplot2)
if (!require('kernlab')) install.packages('kernlab'); library(kernlab)
if (!require('arm')) install.packages('arm'); library(arm)
if (!require('ipred')) install.packages('ipred'); library(ipred)
if (!require('KernelKnn')) install.packages('KernelKnn'); library(KernelKnn)
if (!require('RcppArmadillo')) install.packages('RcppArmadillo'); library(RcppArmadillo)

}

#listWrappers()   # Available models in SuperLearner
set.seed(0-0)

#------------------
# DATA PREPARATION
#------------------
df <- read.csv('data/capstone.dataimp.csv') # data set with Boruta selected fetures
df <- df[-1]

# use 500 observations for developing the model
df <- df[1:10000,]

do.smote <- TRUE

if(do.smote) {

  df$readmitted <- as.factor(df$readmitted)
  table(df$readmitted)

  if (!require('DMwR')) install.packages('DMwR'); library(DMwR)
  df.smote <- SMOTE(readmitted~., df, perc.over=100, perc.under=220)
  table(df.smote$readmitted)

  par(mfrow=c(1,2))
  plot(df['readmitted'],las=1,col='lightblue',xlab='df$readmitted',main='Original')
  plot(df.smote['readmitted'],las=1,col='lightgreen',xlab='df.smote$readmitted',main='SMOTE')
  par(mfrow=c(1,1))

  # When converted from factor to numberic, '0' and '1' become '1' and '2'.
  df.smote$readmitted <- as.numeric(factor(df.smote$readmitted))-1
  #tail(df.smote$readmitted)

  part <- sample(2, nrow(df.smote), replace=TRUE, prob=c(0.7,0.3))

  train <- df.smote[part==1,]
  test  <- df.smote[part==2,]

} else {

  part  <- sample(2, nrow(df), replace=TRUE, prob=c(0.7,0.3))
  train <- df[part==1,]
  test  <- df[part==2,]
}

#names(train)

# Check the index of 'readmitted'
x.train <- train[,-27]
y.train <- train[, 27]

#names(x.train)

x.test  <- test[,-27]
y.test  <- test[, 27]

#names(x.test)

#-----------------------------
# ENSEMBLE & CROSS-VALIDATION
#-----------------------------

# For diabetes data set, the following algorithms commented out if
# low or zero coefficients or compatibility issues in previous test runs.

#---------
# Tuning
#---------
xgboost.custom <- create.Learner('SL.xgboost'
  ,tune=list(
    ntrees=c(500,1000), max_depth=3:4
   ,shrinkage=c(0.01,0.1,0.3), minobspernode=c(10,30)
   )
  ,detailed_names = TRUE, name_prefix = 'xgboost'
)
#environment(xgboost.custom) <-asNamespace("SuperLearner")

ranger.custom <- create.Learner('SL.ranger'
 ,tune = list(
    num.trees = c(500,1000)
   ,mtry = floor(sqrt(ncol(x.train))*c(0.5,1,2))
   #,nodesize = c(1,3,5)
   )
 ,detailed_names = TRUE, name_prefix = 'ranger'
)
#environment(ranger.custom) <-asNamespace("SuperLearner")
if(TRUE){
glmnet.custom <-  create.Learner('SL.glmnet'
 ,tune = list(
   alpha  = seq(0, 1, length.out=10)  # (0,1)=>(ridge, lasso)
  ,nlambda = seq(0, 10, length.out=10)
   )
 ,detailed_names = TRUE, name_prefix = 'glmnet'
)
}
#'
#ranger.custom <- function(...) SL.ranger(...,num.trees=1000, mtry=5)
#kernelKnn.custom <- function(...) SL.kernelKnn(...,transf_categ_cols=TRUE)

#-----------------------
# SuperLearner Settings
#-----------------------
family   <-  'binomial' #'gaussian'
nnls     <- 'method.NNLS'   # NNLS-default
auc      <- 'method.AUC'   # NNLS-default
nnloglik <- 'method.NNloglik'
#SL.algorithm <- list(c('SL.ranger','screen.corP'),c('SL.xgboost','screen.corP'))
SL.algorithm <- c(
  #'SL.ranger','SL.xgboost'
  ranger.custom$names  #,c('ranger.custom$names','screen.corP')
 ,xgboost.custom$names  #,c('xgboost.custom$names','screen.corP')
 #glmnet.custom$names#,'screen.glmnet'
 #,c('screen.randomForest','screen.randomForest')
 #,'SL.xgboost'
 #,'SL.glm'
 #,'SL.bayesglm'
 #,c('kernelKnn.custom','screen.corP')
 #,c('SL.nnet','screen.corP')
 #,c('SL.gbm','screen.corP')
 #,SL.treebag'
 #,'SL.svmRadial'
)

#-------------------------------
# Multicore/Parallel Processing
#-------------------------------

nfold <- 3 # external cross-validation

if (!require('parallel')) install.packages('parallel'); library(parallel)
if (!require('doParallel')) install.packages('doParallel'); library(doParallel)
#if (!require('Rmpi')) install.packages('Rmpi'); library(Rmpi)

cl <- makeCluster(detectCores()-1)

clusterExport(cl, c( listWrappers()

  ,'SuperLearner','CV.SuperLearner','predict.SuperLearner'
  ,'nfold','y.train','x.train','x.test'
  ,'family','nnls','auc','nnloglik'

  ,'SL.algorithm'
  ,ranger.custom$names,xgboost.custom$names
  ,glmnet.custom$names

  ))

clusterSetRNGStream(cl, iseed=135)

## load libraries on workers
clusterEvalQ(cl, library(SuperLearner))
clusterEvalQ(cl, library(ranger))
clusterEvalQ(cl, library(xgboost))
clusterEvalQ(cl, library(caret))
#clusterEvalQ(cl, library(kernlab))
#clusterEvalQ(cl, library(arm))
#clusterEvalQ(cl, library(MASS))
#clusterEvalQ(cl, library(klaR))
#clusterEvalQ(cl, library(nnet))
#clusterEvalQ(cl, library(e1071))
#clusterEvalQ(cl, library(rpart))

system.time({

    # NNLS
    clusterEvalQ(cl, {
      ensem.nnls <- SuperLearner(Y=y.train, X=x.train, verbose=TRUE
        ,family=family,method=nnls
        ,SL.library=SL.algorithm,cvControl=list(V=nfold)
          )
      saveRDS(ensem.nnls, 'ensem.nnls')
    })

    ensem.nnls.cv <- CV.SuperLearner(Y=y.train, X=x.train, verbose=TRUE
      ,parallel=cl
      ,family=family,method=nnls
      ,SL.library=SL.algorithm,cvControl=list(V=nfold)
    )
    saveRDS(ensem.nnls.cv, 'ensem.nnls.cv')

    # AUC
    clusterEvalQ(cl, {
      ensem.auc <- SuperLearner(Y=y.train, X=x.train, verbose=TRUE
        ,family=family,method=nnls
        ,SL.library=SL.algorithm,cvControl=list(V=nfold)
      )
      saveRDS(ensem.auc, 'ensem.auc')
    })

    ensem.auc.cv <- CV.SuperLearner( Y=y.train, X=x.train, verbose=TRUE
      ,parallel=cl
      ,family=family,method=auc
      ,SL.library=SL.algorithm,cvControl=list(V=nfold)
      )
    saveRDS(ensem.auc.cv, 'ensem.auc.cv')

    # NNLogLik
    clusterEvalQ(cl, {
      ensem.nnloglik <- SuperLearner(Y=y.train, X=x.train, verbose=TRUE
        ,family=family,method=nnls
        ,SL.library=SL.algorithm,cvControl=list(V=nfold)
      )
      saveRDS(ensem.nnloglik, 'ensem.nnloglik')
    })

    ensem.nnloglik.cv <- CV.SuperLearner( Y=y.train, X=x.train, verbose=TRUE
      ,parallel=cl
      ,family=family,method=nnloglik
      ,SL.library=SL.algorithm,cvControl=list(V=nfold)
      )
    saveRDS(ensem.nnloglik.cv, 'ensem.nnloglik.cv')

  })

stopCluster(cl)

#------------------------------------------
# Read in results form papallel processing
#------------------------------------------
ensem.nnls <- readRDS('ensem.nnls');ensem.nnls$times;ensem.nnls
ensem.auc  <- readRDS('ensem.auc');ensem.auc$times;ensem.auc
ensem.nnloglik <- readRDS('ensem.nnloglik');ensem.nnloglik$times;ensem.nnloglik

summary(ensem.nnls)
table(simplify2array(ensem.nnls$whichDiscreteSL))
plot(ensem.nnls) + theme_bw()

summary(ensem.auc)
table(simplify2array(ensem.auc$whichDiscreteSL))
plot(ensem.auc) + theme_bw()

summary(ensem.nnloglik)
table(simplify2array(ensem.nnloglik$whichDiscreteSL))
plot(ensem.nnloglik) + theme_bw()

compare.cvRisk <- noquote(cbind(
   ensem.nnls$cvRisk
  ,ensem.auc$cvRisk
  ,ensem.nnloglik$cvRisk
));colnames(compare.cvRisk) <- c(
  'nnls', 'auc','nnloglik'
);compare.cvRisk

#-------------------------
# 2D Scatter Plot (Risks)
#-------------------------
if (!require('dplyr')) install.packages('dplyr'); library(dplyr)
if (!require('plotly')) install.packages('plotly'); library(plotly)

p2d.method.risk <- plot_ly( as.data.frame(compare.cvRisk)
  ,type='scatter',mode = 'markers'
  ,width=1280 ,height=700 #,margin=5
  ,x = ~ensem.nnls$libraryNames, y=~ensem.nnls$cvRisk, name='nnls'
  ,marker = list(
    size = 10, opacity = 0.5
    ,line = list( color = 'black', width = 1)) ) %>%
  add_trace(y = ~ensem.auc$cvRisk, name='auc') %>%
  add_trace(y = ~ensem.nnloglik$cvRisk, name='nnloglik') %>%
  layout( title='Resulted Risks by Library and Method'
          ,xaxis=list(title='library')
          ,yaxis=list(title='risk')
          ,plot_bgcolor="rgb(230,230,230)"
  );p2d.method.risk

#------------
# PREDICTION
#------------
pred.nnls<- predict.SuperLearner(ensem.nnls, x.test, onlySL=TRUE)
#str(pred.nnls);summary(pred.nnls$library.predict);summary(pred.nnls$pred)
pred.auc <- predict.SuperLearner(ensem.auc, x.test, onlySL=TRUE)
pred.nnloglik <- predict.SuperLearner(ensem.nnloglik, x.test, onlySL=TRUE)

#----------------------------------
# Summary of Predictions by Method
#----------------------------------
pred.method.summary <- noquote(cbind(
  summary(pred.nnls$pred)
 ,summary(pred.auc$pred)
 ,summary(pred.nnloglik$pred)
));colnames(pred.method.summary) <- c('nnls','auc','nnloglik'
);pred.method.summary


#----------------------------
# PREDICTION TYPES BY METHOD
#----------------------------
threshold <- 0.5

pred_type <- function(plist, label=y.test, cutoff=0.5) {
  ptype <- rep(NA, length(y.test))
  ptype <-
    ifelse(plist >= cutoff & label == 1, "TP",
      ifelse(plist >= cutoff & label == 0, "FP",
        ifelse(plist < cutoff & label == 1, "FN",
          ifelse(plist < cutoff & label == 0, "TN", '??'))))
  return (ptype)
}

label.description <- ifelse(y.test==0,'Not readmitted',
  ifelse(y.test==1,'Readmitted','??'))

pred.type <- noquote(cbind(
   pred.nnls$pred    , pred_type(pred.nnls$pred     ,y.test ,threshold)
  ,pred.auc$pred     , pred_type(pred.auc$pred      ,y.test ,threshold)
  ,pred.nnloglik$pred, pred_type(pred.nnloglik$pred ,y.test ,threshold)
  ,y.test
  ,label.description
));colnames(pred.type) <- c(
   'nnls','nnls.type'
  ,'auc','auc.type'
  ,'nnloglik','nnloglik.type'
  ,'label','description'
);pred.type

summary(y.test)

#-----------------------------------------------
# 2D scatter Plot of Prediciton Types by Method
#-----------------------------------------------
p2d.method.pred <- plot_ly( as.data.frame(pred.type)
  ,type='scatter',mode = 'markers'
  ,width=1280 ,height=700 #,margin=5
  ,x=1:nrow(pred.type) ,y=~pred.type[,'label'] ,name='label'
  ,marker = list( size = 10 ,opacity = 0.5
     ,line = list( color = 'black' ,width = 1))
) %>% add_trace(y = ~pred.type[,'nnls'] ,name='nnls'
                ,hoverinfo = 'text', text = ~paste(
                  'nnls:' ,pred.type[,'nnls']
                  ,'\nlabel:' ,pred.type[,'description']
                  ,'\nthreshold:' ,threshold
                  ,'\ntype:' ,pred.type[,'nnls.type'])
) %>% add_trace(y = ~pred.type[,'auc'] ,name='auc'
                ,hoverinfo = 'text', text = ~paste(
                  'auc:' ,pred.type[,'auc']
                  ,'\nlabel:' ,pred.type[,'description']
                  ,'\nthreshold:' ,threshold
                  ,'\ntype:' ,pred.type[,'auc.type'])
) %>% add_trace(y = ~pred.type[,'nnloglik'] ,name='nnloglik'
                ,hoverinfo = 'text', text = ~paste(
                  'nnloglik:' ,pred.type[,'nnloglik']
                  ,'\nlabel:' ,pred.type[,'description']
                  ,'\nthreshold:' ,threshold
                  ,'\ntype:' ,pred.type[,'nnloglik.type'])
) %>% layout( title=sprintf("Prediciton & Error Type by Method (Threshold at %.2f)", threshold)
          ,xaxis=list(title='observation')
          ,yaxis=list(title='prediction')
          ,plot_bgcolor='rgb(230,230,230)'
          ,shapes=list( type="line"
            ,line=list( color='black', width=1, name='threshold')
              ,x0=0, x1=nrow(pred.type), xref="paper"
              ,y0=threshold ,y1=threshold)
);p2d.method.pred

auc

#------------------
# CONSUSION MATRIX
#------------------
# Converting probabilities into classification
pred.nnls.converted     <- ifelse(pred.nnls$pred>=0.5,1,0)
pred.auc.converted      <- ifelse(pred.auc$pred>=0.5,1,0)
pred.nnloglik.converted <- ifelse(pred.nnloglik$pred>=0.5,1,0)

# Confusion Matrix
cm.nnls <- confusionMatrix(factor(pred.nnls.converted), factor(y.test));cm.nnls
cat('Mean Square Error (NNLS) = ',mse.nnls <- mean((y.test-pred.nnls$pred)^2))
cm.auc <- confusionMatrix(factor(pred.auc.converted), factor(y.test));cm.auc
cat('Mean Square Error (AUC) = ',mse.auc <- mean((y.test-pred.auc$pred)^2))
cm.nnloglik <- confusionMatrix(factor(pred.nnloglik.converted), factor(y.test));cm.nnloglik
cat('Mean Square Error (NNLogLik) = ',mse.nnloglik <- mean((y.test-pred.nnloglik$pred)^2))

noquote(cbind(
  Method=c('nnls','auc','nnloglik')
 ,MSE=c(mse.nnls,mse.auc,mse.nnloglik)
 ,Accuracy=c(
    cm.nnls$overall['Accuracy']
   ,cm.auc$overall['Accuracy']
   ,cm.nnloglik$overall['Accuracy']
  )
))

#------------------------------------------------
# COMPARING PREDICTONS MADE BY THE THREE METHODS
#------------------------------------------------
if (!require('dplyr')) install.packages('dplyr'); library(dplyr)
if (!require('plotly')) install.packages('plotly'); library(plotly)

pred3 <- as.data.frame( cbind(
   pred.nnls$pred
  ,pred.auc$pred
  ,pred.nnloglik$pred
));pred3$label <- as.factor(ifelse(y.test == 0,'Not readmitted','Readmitted')
);pred3$y.test <- y.test
colnames(pred3) <- c('nnls','auc','nnloglik','label','y.test');pred3


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

#-------------------------------
# 2D Scatter Plot (Predictions)
#-------------------------------
p2d.pred.method <- plot_ly( pred3
  ,type='scatter',mode = 'markers-lines',lwd=0.1
  ,width=1280 ,height=700 #,margin=5
  ,x = ~(1:length(pred3$label)), y=~pred3$nnls, name='NNLS' ) %>%
  add_trace(y = ~pred3$auc, name='AUC') %>%
  add_trace(y = ~pred3$nnloglik, name='NNLokLik') %>%
  add_trace(y = ~pred3$y.test, name='Label') %>%
  layout( title='Comparing Predictions Made by NNLS/AUC/NNLogLik'
          ,xaxis=list(title='')
          ,yaxis=list(title='Prediction')
          ,plot_bgcolor="rgb(230,230,230)"
  );p2d.pred.method

#-----------------
# 3D Scatter Plot
#-----------------
col_ramp <- colorRampPalette(c("green","orange","red","black"))(100)

p3d.pred.method <- plot_ly( pred3
  ,x = ~pred3$auc, y = ~pred3$nnls, z = ~pred3$nnloglik, color = pred3$label
  ,hoverinfo = 'text'
  ,text = ~paste(
     'auc:\t',round(pred3$auc,7)
    ,'\nnnls:\t', round(pred3$nnls,7)
    ,'\nnnloglik:\t', round(pred3$nnloglik,7)
    ,'\nlabel:\t', pred3$label)
  ,colors = c('blue', 'yellow', 'red')
  ,marker = list(
     size = 10, opacity = 0.5
    ,line = list( color = 'black', width = 1))
) %>% add_markers() %>%
  layout( title='Comparing Predictions Made by NNLS/AUC/NNLogLik',scene = list(
    xaxis = list(
      title = 'AUC'
      ,backgroundcolor="rgb(204,204,204)"
      ,gridcolor="rgb(255, 255, 255)"
      ,zerolinecolor="rgb(0,0,0)"
      ,showbackground=TRUE)
    ,yaxis = list(
      title = 'NNLS'
      ,backgroundcolor="rgb(217,217,217)"
      ,gridcolor="rgb(255, 255, 255)"
      ,zerolinecolor="rgb(0,0,0)"
      ,showbackground=TRUE)
    ,zaxis = list(
      title = 'NNLogLik'
      ,backgroundcolor="rgb(230,230,230)"
      ,gridcolor="rgb(255, 255, 255)"
      ,zerolinecolor="rgb(0,0,0)"
      ,showbackground=TRUE)
    ,camera = list(
      up=list(x=0, y=0, z=1)
      ,center=list(x=0, y=0, z=0)
      ,eye=list(x=2, y=0.4, z=0.25)
    )
  )
  );p3d.pred.method

#---------------------------------------------
# External Cross-Validation for the Ensembles
#---------------------------------------------
readRDS('ensem.nnls.cv');summary(ensem.nnls.cv)
plot(ensem.nnls.cv) + theme_minimal()

readRDS('ensem.auc.cv');summary(ensem.auc.cv)
plot(ensem.auc.cv) + theme_minimal()

readRDS('ensem.nnloglik.cv');summary(ensem.nnloglik.cv)
plot(ensem.nnloglik.cv) + theme_minimal()

#----
#AUC
#----
auc <- cvAUC(p1,labels=y.test)$cvAUC;auc
auc <- cvAUC(p1,labels=y.test)$cvAUC;auc

#Plot fold AUCs
plot(auc$perf, col="grey82", lty=3, main="10-fold CV AUC")
#Plot CV AUC
plot(auc$perf, col="red", avg="vertical", add=TRUE)
