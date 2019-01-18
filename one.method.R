#-------------------------
# 2D Scatter Plot (Risks)
#-------------------------
if (!require('dplyr')) install.packages('dplyr'); library(dplyr)
if (!require('plotly')) install.packages('plotly'); library(plotly)

(p2d.method.risk <- plot_ly( as.data.frame(cbind(ensem.nnls$cvRisk,ensem.nnls$coef))
  ,type='scatter',mode = 'markers'
  ,width=1600 ,height=900 #,margin=5
  ,x=~ensem.nnls$libraryNames ,y = ~ensem.nnls$cvRisk, name='risk'
  ,hoverinfo = 'text' ,text = ~paste(
    'func:' ,ensem.nnls$libraryNames
   ,'\nrisk:' ,ensem.nnls$cvRisk)
  ,marker=list( size = 10, opacity = 0.5
                ,line=list( color='black' ,width=1
                #,shape='spline' ,smoothing=1.3
                ))
) %>% add_trace(y = ~ensem.nnls$coef, name='coefficient',yaxis='y2'
                ,hoverinfo = 'text' ,text = ~paste(
                  'func:' ,ensem.nnls$libraryNames
                  ,'\ncoef:' ,ensem.nnls$coef)
) %>% layout( title=sprintf(
   "Ricks and Coefficients Based on Tuning Parameters (SMOTE'd Training Data = %i obs.)", nrow(x.train))
  ,xaxis=list(title='')
  ,yaxis=list(title='risk'
              #,range=c(min(ensem.nnls$cvRisk),max(ensem.nnls$cvRisk)+0.1)
              )
  ,yaxis2=list(title='coefficient' #
               #,range=c(min(ensem.nnls$coef),max(ensem.nnls$coef)+0.1)
               ,overlaying='y' ,side='right')
  ,margin=list() #l=50, r=50, b=50, t=50, pad=4
  ,plot_bgcolor='rgb(250,250,250)'
))
#%>% add_lines(y = ~ensem.auc$cvRisk, colors = "black", alpha = 0.2)


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

  ,y.test ,ifelse(y.test==0 ,'not readmitted','readmitted')

));colnames(pred.type) <- c(
  'nnls'     ,'nnls.type'     ,'nnls.prediction'
#  ,'auc'      ,'auc.type'      ,'auc.prediction'
#  ,'nnloglik' ,'nnloglik.type' ,'nnloglik.prediction'
  ,'label'    ,'description'
);pred.type




#-----------------------------------------------
# 2D scatter Plot of Prediciton Types by Method
#-----------------------------------------------
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
) %>% layout( title=sprintf(
  'Predicitons & Error Types by Methods (Threshold = %.2f)', threshold)
  ,xaxis=list(title='observation')
  ,yaxis=list(title='prediction')
  ,plot_bgcolor='rgb(230,230,230)'
  ,annotations=list( text=''  # legend title
                     ,yref='paper',xref='paper'
                     ,y=1.025 ,x=1.09 ,showarrow=FALSE)
  ,shapes=list( type='line'
                ,x0=0, x1=1 ,xref = 'paper'
                ,y0=threshold ,y1=threshold
                ,line=list( color='black', width=1)
  )
);p2d.method.pred


