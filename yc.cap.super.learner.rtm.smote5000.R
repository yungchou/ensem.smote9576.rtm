if (!require('munsell')) install.packages("munsell"); library(munsell)
if (!require('ggplot2')) install.packages('ggplot2'); library(ggplot2)
if (!require('SuperLearner')) install.packages('SuperLearner'); library(SuperLearner)
if (!require('ranger' )) install.packages('ranger' ); library(ranger )
if (!require('xgboost')) install.packages('xgboost'); library(xgboost)
if (!require('caret'  )) install.packages('caret'  ); library(caret  )

if(!TRUE){ # skip the following

if (!require('mlbench')) install.packages('mlbench'); library(mlbench)
if (!require('rpart')) install.packages('rpart'); library(rpart)

if (!require('h2o')) install.packages('h2o'); library(h2o)

  install.packages("devtools");library(devtools
);install_github("AppliedDataSciencePartners/xgboostExplainer",force = TRUE)

if (!require('MASS')) install.packages('MASS'); library(MASS)
if (!require('cvAUC')) install.packages('cvAUC'); library(cvAUC)
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
df <- df[1:20000,]

do.smote <- TRUE

if(do.smote) {

  df$readmitted <- as.factor(df$readmitted)
  table(df$readmitted)

  if (!require('DMwR')) install.packages('DMwR'); library(DMwR)
  df.smote <- SMOTE(readmitted~., df, perc.over=100, perc.under=220)
  table(df.smote$readmitted)

  par(mfrow=c(1,2)
  );plot(df['readmitted'],las=1,col='lightblue'
         ,xlab='label(readmitted)',main=sprintf('Imported Data Set\n(%i obs.)',nrow(df))
  );plot(df.smote['readmitted'],las=1,col='lightgreen'
         ,xlab='label(readmitted)',main=sprintf('Employed SMOTEd Data\n(%i obs.)',nrow(df.smote))
  );par(mfrow=c(1,1))

  # When converted from factor to numberic, '0' and '1' become '1' and '2'.
  df.smote$readmitted <- as.numeric(factor(df.smote$readmitted))-1
  #tail(df.smote$readmitted)

  total.obs <- nrow(df.smote)

  part <- sample(3 ,total.obs ,replace=TRUE ,prob=c(0.6,0.2,0.2))
  train <- df.smote[part==1,]
  test  <- df.smote[part==2,]
  hold  <- df.smote[part==3,]  # for cross validation

} else {

  total.obs <- nrow(df.smote)

  part  <- sample(3 ,total.obs ,replace=TRUE ,prob=c(0.6,0.2,0.2))
  train <- df[part==1,]
  test  <- df[part==2,]
  hold  <- df[part==3,]  # for cross validation
}

cat('Total observations = ',total.obs
);data.usage <- noquote(cbind(
   c('train','test','hold')
  ,c(nrow(train),nrow(test),nrow(hold))
  ));colnames(data.usage) <- c('usage','obs.');data.usage

# Check the index of 'readmitted'
x.train <- train[,-27]
y.train <- train[, 27]

x.test  <- test[,-27]
y.test  <- test[, 27]

x.hold  <- hold[,-27]
y.hold  <- hold[, 27]

#---------
# Tuning
#---------
xgboost.custom <- create.Learner('SL.xgboost'
  ,tune=list(
    ntrees=c(500,1000) ,max_depth=4   # 1:4
   ,shrinkage=c(0.01,0.1) ,minobspernode=c(10,30)
   )
  ,detailed_names = TRUE ,name_prefix = 'xgboost'
)

ranger.custom <- create.Learner('SL.ranger'
 ,tune = list(
    num.trees = c(1000,1500)
   ,mtry = floor(sqrt(ncol(x.train))*c(1,2))
   )
 ,detailed_names = TRUE ,name_prefix = 'ranger'
)

if(!TRUE){ # !TRUE to skip
glmnet.custom <-  create.Learner('SL.glmnet'
 ,tune = list(
   alpha  = seq(0 ,1 ,length.out=10)  # (0,1)=>(ridge, lasso)
  ,nlambda = seq(0 ,10 ,length.out=10)
   )
 ,detailed_names = TRUE ,name_prefix = 'glmnet'
)}

#ranger.custom <- function(...) SL.ranger(...,num.trees=1000, mtry=5)
#kernelKnn.custom <- function(...) SL.kernelKnn(...,transf_categ_cols=TRUE)

#-----------------------
# SuperLearner Settings
#-----------------------
family   <- 'binomial' #'gaussian'
nnls     <- 'method.NNLS' # NNLS-default
auc      <- 'method.AUC'
nnloglik <- 'method.NNloglik'

SL.algorithm <- c(
  # baseline
  #'SL.ranger'  # regression & classification
 #,'SL.xgboost' # regression & classification
  # tuning
 ranger.custom$names
 ,xgboost.custom$names
 #,glmnet.custom$names
 #,'SL.earth' # regression
 #,'SL.glm'
 #,'SL.bayesglm'
 #,c('SL.kernelKnn','screen.corP')
 #,c('SL.nnet','screen.corP')
 #,c('SL.gbm','screen.corP')
 #,'SL.treebag'
 #,'SL.svm','SL.svmRadial'
)

#-------------------------------
# Multicore/Parallel Processing
#-------------------------------

nfold <- 5 # external cross-validation

if (!require('parallel')) install.packages('parallel'); library(parallel)
if (!require('doParallel')) install.packages('doParallel'); library(doParallel)
#if (!require('Rmpi')) install.packages('Rmpi'); library(Rmpi)

cl <- makeCluster(detectCores()-1)

clusterExport(cl, c( listWrappers()

  ,'SuperLearner','CV.SuperLearner','predict.SuperLearner'
  ,'nfold','x.train','y.train','x.test','y.test','x.hold','y.hold'
  ,'family','nnls','auc','nnloglik'

  ,'SL.algorithm'
  ,ranger.custom$names,xgboost.custom$names
#  ,glmnet.custom$names

  ))

clusterSetRNGStream(cl, iseed=135)

## Load libraries on workers
clusterEvalQ(cl, {
  library(SuperLearner);library(caret)
  library(ranger);library(xgboost)
#library(kernlab)
#library(arm)
#ibrary(MASS)
#library(klaR)
#ibrary(nnet)
#library(e1071)
})

clusterEvalQ(cl, {

  ensem.nnls <- SuperLearner(Y=y.train ,X=x.train ,verbose=TRUE
    ,family=family ,method=nnls ,SL.library=SL.algorithm
    );saveRDS(ensem.nnls ,'ensem.nnls')

  ensem.auc <- SuperLearner(Y=y.train ,X=x.train ,verbose=TRUE
    ,family=family ,method=auc ,SL.library=SL.algorithm
    );saveRDS(ensem.auc ,'ensem.auc')

  ensem.nnloglik <- SuperLearner(Y=y.train ,X=x.train ,verbose=TRUE
   ,family=family ,method=nnloglik ,SL.library=SL.algorithm
   );saveRDS(ensem.nnloglik ,'ensem.nnloglik')

})

system.time({
  ensem.nnls.cv <- CV.SuperLearner(Y=y.hold ,X=x.hold ,verbose=TRUE
    ,cvControl=list(V=nfold) ,parallel=cl
    ,family=family ,method=nnls ,SL.library=SL.algorithm
    );saveRDS(ensem.nnls.cv ,'ensem.nnls.cv')
  })

system.time({
  ensem.auc.cv <- CV.SuperLearner( Y=y.hold ,X=x.hold ,verbose=TRUE
    ,cvControl=list(V=nfold) ,parallel=cl
    ,family=family ,method=auc ,SL.library=SL.algorithm
    );saveRDS(ensem.auc.cv ,'ensem.auc.cv')
  })

system.time({
  ensem.nnloglik.cv <- CV.SuperLearner( Y=y.hold ,X=x.hold ,verbose=TRUE
    ,cvControl=list(V=nfold) ,parallel=cl
    ,family=family ,method=nnloglik ,SL.library=SL.algorithm
    );saveRDS(ensem.nnloglik.cv ,'ensem.nnloglik.cv')
  })

stopCluster(cl)

#------------------------------------------
# Read in results form papallel processing
#------------------------------------------
ensem.nnls     <- readRDS('ensem.nnls')    ;ensem.nnls$times    ;ensem.nnls
ensem.auc      <- readRDS('ensem.auc')     ;ensem.auc$times     ;ensem.auc
ensem.nnloglik <- readRDS('ensem.nnloglik');ensem.nnloglik$times;ensem.nnloglik

#----------------------
# Risk and Coefficient
#----------------------
compare <- noquote(cbind(
   ensem.nnls$cvRisk,ensem.nnls$coef
  ,ensem.auc$cvRisk,ensem.auc$coef
  ,ensem.nnloglik$cvRisk,ensem.nnloglik$coef
));colnames(compare) <- c(
   'nnls.cvRisk','nnls.coef'
  ,'auc.cvRisk','auc.coef'
  ,'nnloglik.cvRisk','nnloglik.coef'
);compare[,c(1,3,5)];compare[,c(2,4,6)]

# 2D Scatter Plot (Risks)

if (!require('dplyr')) install.packages('dplyr'); library(dplyr)
if (!require('plotly')) install.packages('plotly'); library(plotly)

(p2d.method.risk <- plot_ly( as.data.frame(compare)
  ,type='scatter',mode = 'markers'
  ,width=1280 ,height=700 #,margin=5
  ,x=~ensem.nnls$libraryNames, y=~compare[,'nnls.cvRisk'], name='nnls'
  ,marker=list( size = 10, opacity = 0.5
    ,line=list( color='black' ,width=1
     #,shape='spline' ,smoothing=1.3
      ))
) %>% add_trace(y = ~compare[,'auc.cvRisk'], name='auc'
) %>% add_trace(y = ~compare[,'nnloglik.cvRisk'], name='nnloglik'
) %>% layout( title=sprintf("Learner/Algorithm Evaluations with Tuning Parameters (SMOTE'd Training Data = %i obs.)", nrow(x.train))
          ,xaxis=list(title='')
          ,yaxis=list(title='risk')
          ,plot_bgcolor='rgb(230,230,230)'
))

(p2d.method.risk <- plot_ly( as.data.frame(compare)
  ,type='scatter',mode = 'markers'
  ,width=1280 ,height=700 #,margin=5
  ,x=~ensem.nnls$libraryNames, y=~compare[,'nnls.coef'], name='nnls'
  ,marker=list( size = 10, opacity = 0.5
    ,line=list( color='black' ,width=1
    #,shape='spline' ,smoothing=1.3
    ))
) %>% add_trace(y = ~compare[,'auc.coef'], name='auc'
) %>% add_trace(y = ~compare[,'nnloglik.coef'], name='nnloglik'
) %>% layout( title=sprintf("Learner/Algorithm Coefficients with Tuning Parameters (SMOTE'd Training Data = %i obs.)", nrow(x.train))
  ,xaxis=list(title='')
  ,yaxis=list(title='risk')
  ,plot_bgcolor='rgb(230,230,230)'
))

#%>% add_lines(y = ~ensem.auc$cvRisk, colors = "black", alpha = 0.2)

#------------
# PREDICTION
#------------
pred.nnls     <- predict.SuperLearner(ensem.nnls     ,x.test ,onlySL=TRUE)
#str(pred.nnls);summary(pred.nnls$library.predict);summary(pred.nnls$pred)
pred.auc      <- predict.SuperLearner(ensem.auc      ,x.test ,onlySL=TRUE)
pred.nnloglik <- predict.SuperLearner(ensem.nnloglik ,x.test ,onlySL=TRUE)

#----------------------------------
# Summary of Predictions by Method
#----------------------------------
pred.method.summary <- noquote(cbind(
  summary(pred.nnls$pred)
 ,summary(pred.auc$pred)
 ,summary(pred.nnloglik$pred)
));colnames(pred.method.summary) <- c('nnls','auc','nnloglik'
);pred.method.summary

#if (!require('ck37r')) install.packages('ck37r') ;library(ck37r)
#plot.SuperLearner(ensem.nnls, Y=y.train, constant = qnorm(0.975),sort = TRUE)

#----------------------------
# PREDICTION TYPES BY METHOD
#----------------------------
cat('Classification threshold:', threshold <- 0.5)

pred_type <- function(plist, label=y.test, cutoff=0.5) {

  ptype <- rep(NA, length(y.test))
  ptype <-
    ifelse(plist >= cutoff & label == 1, "TP",
      ifelse(plist >= cutoff & label == 0, "FP",
        ifelse(plist < cutoff & label == 1, "FN",
          ifelse(plist < cutoff & label == 0, "TN", '??'))))
  return (ptype)
}

pred.type <- noquote(cbind(

   pred.nnls$pred ,pred_type(pred.nnls$pred ,y.test ,threshold)
  ,ifelse(pred.nnls$pred < threshold ,'not readmitted','readmitted')

  ,pred.auc$pred ,pred_type(pred.auc$pred ,y.test ,threshold)
  ,ifelse(pred.auc$pred < threshold ,'not readmitted','readmitted')

  ,pred.nnloglik$pred ,pred_type(pred.nnloglik$pred ,y.test ,threshold)
  ,ifelse(pred.nnloglik$pred < threshold ,'not readmitted','readmitted')

  ,y.test ,ifelse(y.test==0 ,'not readmitted','readmitted')

));colnames(pred.type) <- c(
   'nnls'     ,'nnls.type'     ,'nnls.prediction'
  ,'auc'      ,'auc.type'      ,'auc.prediction'
  ,'nnloglik' ,'nnloglik.type' ,'nnloglik.prediction'
  ,'label'    ,'description'
);pred.type

#summary(y.test)

#----------------------------------------------------
# 2D scatter Plot of Prediciton by Ensemble Learning
#----------------------------------------------------
library(RColorBrewer);library(plotly)

p2d.method.pred <- plot_ly( as.data.frame(pred.type)
  ,x = ~1:nrow(pred.type)
  ,y = ~pred.type[,'label'] ,name='label'
  ,hoverinfo = 'text' ,text = ~paste(
      'label:' ,pred.type[,'label']
     ,'\ndescription:' ,pred.type[,'description'])
  ,type='scatter' ,width=1280 ,height=700 #,margin=5
  ,mode = 'markers+lines'
  ,marker = list( size = 10 ,opacity = 0.5
     #,color = pred.type ,colorbar=list(title = "Viridis")
     #,color = colorRampPalette(brewer.pal(12,'Set1'))(2000)
     ,line = list( color = 'black' ,width = 1))
) %>% add_trace(y = ~pred.type[,'nnls'] ,name='nnls' ,mode = 'markers'
                ,hoverinfo = 'text' ,text = ~paste(
                   'nnls:' ,pred.type[,'nnls']
                  ,'\nthreshold:' ,threshold
                  ,'\nprediction:' ,pred.type[,'nnls.prediction']
                  ,'\n-------------------------------------------'
                  ,'\nlabel:' ,pred.type[,'description']
                  ,'\n-------------------------------------------'
                  ,'\ntype:' ,pred.type[,'nnls.type'])
) %>% add_trace(y = ~pred.type[,'auc'] ,name='auc' ,mode = 'markers'
                ,hoverinfo = 'text', text = ~paste(
                   'auc:' ,pred.type[,'auc']
                  ,'\nthreshold:' ,threshold
                  ,'\nprediction:' ,pred.type[,'auc.prediction']
                  ,'\n-------------------------------------------'
                  ,'\nlabel:' ,pred.type[,'description']
                  ,'\n-------------------------------------------'
                  ,'\ntype:' ,pred.type[,'auc.type'])
) %>% add_trace(y = ~pred.type[,'nnloglik'] ,name='nnloglik' ,mode = 'markers'
                ,hoverinfo = 'text', text = ~paste(
                   'nnloglik:' ,pred.type[,'nnloglik']
                  ,'\nthreshold:' ,threshold
                  ,'\nprediction:' ,pred.type[,'nnloglik.prediction']
                  ,'\n-------------------------------------------'
                  ,'\nlabel:' ,pred.type[,'description']
                  ,'\n-------------------------------------------'
                  ,'\ntype:' ,pred.type[,'nnloglik.type'])
) %>% layout( title=sprintf(
  'Predicitons by Ensemble Learning (Threshold = %.2f)', threshold)
  ,xaxis=list(title='observation')
  ,yaxis=list(title='prediction')
  ,plot_bgcolor='rgb(250,250,250)'
  ,annotations=list( text=''  # legend title
  ,yref='paper',xref='paper'
  ,y=1.025 ,x=1.09 ,showarrow=FALSE)
) %>% add_trace( name='threshold'  #,showlegend=FALSE
  ,y= threshold
  ,marker = list( size = 1
    ,line = list( color = 'black' ,width = 1))
  ,hoverinfo = 'text' ,text=sprintf('threshold = %.2f', threshold)
);p2d.method.pred

#-----------------
# 3D Scatter Plot
#-----------------
p3d.method.pred <- plot_ly( as.data.frame(pred.type)
   ,x = ~pred.type[,'auc'], y = ~pred.type[,'nnls'], z = ~pred.type[,'nnloglik']
   ,hoverinfo = 'text' ,text = ~paste(
      'label:'         ,pred.type[,'label']
     ,'\ndescription:' ,pred.type[,'description']
     ,'\n-------------------------------------------'
     ,'\nthreshold:'   ,threshold
     ,'\n-------------------------------------------'
     ,'\nnnls:'        ,pred.type[,'nnls']
     ,'\nprediction:'  ,pred.type[,'nnls.prediction']
     ,'\ntype:'        ,pred.type[,'nnls.type']
     ,'\n-------------------------------------------'
     ,'\nauc:'        ,pred.type[,'auc']
     ,'\nprediction:' ,pred.type[,'auc.prediction']
     ,'\ntype:'       ,pred.type[,'auc.type']
     ,'\n-------------------------------------------'
     ,'\nnnloglik:'   ,pred.type[,'nnloglik']
     ,'\nprediction:' ,pred.type[,'nnloglik.prediction']
     ,'\ntype:'       ,pred.type[,'nnloglik.type']
      )
     #,colors = c('blue', 'yellow', 'red')
     ,marker = list(size = 10 ,opacity = 0.5
        ,color = colorRampPalette(brewer.pal(12,'Set3'))(2000)
        ,line = list( color = 'black' ,width = 0.5))
) %>% add_markers(
) %>% layout( title=sprintf(
  'Predictions by Ensemble Learning (Threshold = %.2f)', threshold)
  ,scene = list(
    xaxis = list(title='auc'
      ,backgroundcolor='rgb(240,240,240)'
      ,gridcolor='rgb(180, 180, 180)'
      ,zerolinecolor='rgb(0,0,0)'
      ,showbackground=TRUE)
      ,yaxis = list(title='nnls'
        ,backgroundcolor='rgb(240,240,240)'
        ,gridcolor='rgb(180, 180, 180)'
        ,zerolinecolor='rgb(0,0,0)'
        ,showbackground=TRUE)
      ,zaxis = list(title='nnloglik'
        ,backgroundcolor='rgb(240,240,240)'
        ,gridcolor='rgb(180, 180, 180)'
        ,zerolinecolor='rgb(0,0,0)'
        ,showbackground=TRUE)
      ,camera = list(
         up=list(x=0, y=0, z=1)
        ,center=list(x=0, y=0, z=0)
        ,eye=list(x=1.75, y=-1.25, z=0.5)  )
));p3d.method.pred

#------------------
# CONSUSION MATRIX
#------------------
# Converting probabilities into classification
pred.type$nnls.converted     <- ifelse(pred.nnls$pred     >= threshold,1,0)
pred.type$auc.converted      <- ifelse(pred.auc$pred      >= threshold,1,0)
pred.type$nnloglik.converted <- ifelse(pred.nnloglik$pred >= threshold,1,0)

# Confusion Matrix
cm.nnls <- confusionMatrix(factor(pred.type$nnls.converted ), factor(y.test));cm.nnls
cat('Mean Square Error (NNLS) = ',mse.nnls <- mean((y.test-pred.nnls$pred)^2))

cm.auc <- confusionMatrix(factor(pred.type$auc.converted), factor(y.test));cm.auc
cat('Mean Square Error (AUC) = ',mse.auc <- mean((y.test-pred.auc$pred)^2))

cm.nnloglik <- confusionMatrix(factor(pred.type$nnloglik.converted ), factor(y.test));cm.nnloglik
cat('Mean Square Error (NNloglik) = ',mse.nnloglik <- mean((y.test-pred.nnloglik$pred)^2))

pred.accuracy <- noquote(cbind(
  MSE=c(mse.nnls,mse.auc,mse.nnloglik)
 ,Accuracy=c(
    cm.nnls$overall['Accuracy']
   ,cm.auc$overall['Accuracy']
   ,cm.nnloglik$overall['Accuracy']
  )
));rownames(pred.accuracy) <- c('nnls','auc','nnloglik');pred.accuracy

#-------------------------
# CROSS VALIDATON OBJECTS
#-------------------------
ensem.nnls.cv <- readRDS('ensem.nnls.cv');ensem.nnls.cv
ensem.auc.cv  <- readRDS('ensem.auc.cv');ensem.auc.cv
ensem.nnloglik.cv <- readRDS('ensem.nnloglik.cv');ensem.nnloglik.cv

summary(ensem.nnls.cv)
table(simplify2array(ensem.nnls.cv$whichDiscreteSL))

summary(ensem.auc.cv)
table(simplify2array(ensem.auc.cv$whichDiscreteSL))

summary(ensem.nnloglik.cv)
table(simplify2array(ensem.nnloglik.cv$whichDiscreteSL))

#----------
# Stacking
#----------
ensem.nnls.cv.stacking <- plot(ensem.nnls.cv)+theme_bw();ensem.nnls.cv.stacking
ensem.auc.cv.stacking <- plot(ensem.auc.cv)+theme_bw();ensem.auc.cv.stacking
ensem.nnloglik.cv.stacking <- plot(ensem.nnloglik.cv)+theme_bw();ensem.nnloglik.cv.stacking

#----------------------------------------------------------
# CROSS VALIDATION - ROC CURVE
# (receiver operating characteristic curve)
#
# It plots TPR vs. FPR at different classification thresholds.
# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
#----------------------------------------------------------
if (!require('ck37r')) install.packages('ck37r') ;library(ck37r)
ensem.nnls.cv.roc <- cvsl_plot_roc(ensem.nnls.cv);ensem.nnls.cv.roc
ensem.auc.cv.roc <- cvsl_plot_roc(ensem.auc.cv);ensem.auc.cv.roc
ensem.nnloglik.cv.roc <- cvsl_plot_roc(ensem.nnloglik.cv);ensem.nnloglik.cv.roc

#----------------------------------------------------------
# CROSS VALIDATION - AUC
# (Area under the ROC Curve)
#
# AUC provides an aggregate measure of performance across
# all possible classification thresholds. Consider AUC as
# the probability that the model ranks a random positive
# example more highly than a random negative example.
#
# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
#----------------------------------------------------------
if (!require('ck37r')) install.packages('ck37r') ;library(ck37r)

cvsl_weights(ensem.nnls.cv)
cvsl_weights(ensem.auc.cv)
cvsl_weights(ensem.nnloglik.cv)

cvsl_auc(ensem.nnls.cv)
cvsl_auc(ensem.auc.cv)
cvsl_auc(ensem.nnloglik.cv)
