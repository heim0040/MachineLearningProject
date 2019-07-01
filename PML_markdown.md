---
title: "PML course project"
author: "Matt Heimdahl"
date: "6/30/2019"
output: 
  html_document:
    keep_md: yes
---



## Executive Summary

In this analysis we explore data from the Weight Lifting Exercises Dataset, a collection of data from accelerometers from 6 participants who were asked to perform barbell lifts both correctly and incorrectly in 5 different patterns. This study is from the field of human activity recognition (HAR) research, which aims to quantify how well an activity is performed.

## Exploring the Data

We read in the training and testing data available online. Our outcome variable, "classe", measures the 5 different weightlifting patterns described above. The testing dataset does not include a "classe" variable, but will predict "classe" for the individuals in the testing dataset later in our analysis.


```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

## Data Hygiene

There are several ID and timestamp variables that we will remove from the datasets. We check for missing variables and discover that several variables have a significant portion of NAs (97% in the training and 100% in the testing data) so we will remove these variables from our analysis. We also test variables to see if they only have one or few unique values and eliminate these from our dataset.


```r
# remove columns
training <- select(training, -c(1:6))
testing <- select(testing, -c(1:6))

# check missing values
table(is.na(training))
```

```
## 
##   FALSE    TRUE 
## 1734316 1287472
```

```r
tr <- sapply(training, function(x) sum(is.na(x))/length(x))*100

table(is.na(testing))
```

```
## 
## FALSE  TRUE 
##  1080  2000
```

```r
te <- sapply(testing, function(x) sum(is.na(x))/length(x))*100

# remove columns with any NAs
not_any_na <- function(x) all(!is.na(x))
training2 <- training %>% select_if(not_any_na)
dim(training2)
```

```
## [1] 19622    87
```

```r
# remove zero covariates
nzv <- nearZeroVar(training2,saveMetrics=TRUE)

nzv <- nearZeroVar(training2)
training2 <- training2[, -nzv]
dim(training2)
```

```
## [1] 19622    54
```

## Model Fitting

Next, we fit our model. For this analysis we use a Gradient Boosting Model (GBM) using the caret package.
We will use cross-validation in the model to obtain a estimate of the accuracy with the test set.


```r
# parallel processing
library(parallel)
library(doParallel)
```

```
## Loading required package: foreach
```

```
## Loading required package: iterators
```

```r
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)

# configure trainControl
trainControl <- trainControl(method="cv", number=5, allowParallel=TRUE, savePredictions = T)

# seperate out outcome variable and predictors for the model
x <- training2[,-54]
y <- training2[,54]

# boosting model
set.seed(831)
gbmFit <- caret::train(x, y, method="gbm", trControl=trainControl, verbose=FALSE)
print(gbmFit)
```

```
## Stochastic Gradient Boosting 
## 
## 19622 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15698, 15698, 15698, 15697, 15697 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7605749  0.6962972
##   1                  100      0.8318723  0.7871536
##   1                  150      0.8725410  0.8386713
##   2                   50      0.8864540  0.8562159
##   2                  100      0.9440932  0.9292596
##   2                  150      0.9672304  0.9585329
##   3                   50      0.9337477  0.9161351
##   3                  100      0.9745183  0.9677575
##   3                  150      0.9890937  0.9862037
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
# stop parallel processing
stopCluster(cluster)
registerDoSEQ()
```

## Model Results

The confusion matrix indicates that the model accurately predicts over 98% of the training cases. The resample results show that the accuracy is at the same relative level in all 5 of the partitions used in cross validation.



```r
# confusion matrix
gbmFit
```

```
## Stochastic Gradient Boosting 
## 
## 19622 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15698, 15698, 15698, 15697, 15697 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7605749  0.6962972
##   1                  100      0.8318723  0.7871536
##   1                  150      0.8725410  0.8386713
##   2                   50      0.8864540  0.8562159
##   2                  100      0.9440932  0.9292596
##   2                  150      0.9672304  0.9585329
##   3                   50      0.9337477  0.9161351
##   3                  100      0.9745183  0.9677575
##   3                  150      0.9890937  0.9862037
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
gbmFit$resample
```

```
##    Accuracy     Kappa Resample
## 1 0.9908280 0.9884002    Fold4
## 2 0.9872579 0.9838791    Fold3
## 3 0.9913354 0.9890385    Fold2
## 4 0.9900637 0.9874328    Fold5
## 5 0.9859837 0.9822681    Fold1
```

```r
confusionMatrix.train(gbmFit)
```

```
## Cross-Validated (5 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.1  0.0  0.0  0.0
##          B  0.0 19.0  0.1  0.1  0.1
##          C  0.0  0.2 17.3  0.2  0.0
##          D  0.0  0.0  0.1 16.1  0.2
##          E  0.0  0.0  0.0  0.0 18.1
##                             
##  Accuracy (average) : 0.9891
```

```r
# look at tuning results and final model
gbmFit$bestTune
```

```
##   n.trees interaction.depth shrinkage n.minobsinnode
## 9     150                 3       0.1             10
```

```r
plot(gbmFit)
```

![](PML_markdown_files/figure-html/results-1.png)<!-- -->

```r
gbmFit$results
```

```
##   shrinkage interaction.depth n.minobsinnode n.trees  Accuracy     Kappa
## 1       0.1                 1             10      50 0.7605749 0.6962972
## 4       0.1                 2             10      50 0.8864540 0.8562159
## 7       0.1                 3             10      50 0.9337477 0.9161351
## 2       0.1                 1             10     100 0.8318723 0.7871536
## 5       0.1                 2             10     100 0.9440932 0.9292596
## 8       0.1                 3             10     100 0.9745183 0.9677575
## 3       0.1                 1             10     150 0.8725410 0.8386713
## 6       0.1                 2             10     150 0.9672304 0.9585329
## 9       0.1                 3             10     150 0.9890937 0.9862037
##    AccuracySD     KappaSD
## 1 0.011953433 0.015240719
## 4 0.002621464 0.003372148
## 7 0.005814796 0.007366762
## 2 0.005477650 0.006972249
## 5 0.004490405 0.005694407
## 8 0.003977070 0.005035811
## 3 0.006390160 0.008079110
## 6 0.004865891 0.006170191
## 9 0.002346071 0.002969175
```

```r
gbmFit$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 53 had non-zero influence.
```

```r
# look at performance per partition and resample
gbmFit$resample
```

```
##    Accuracy     Kappa Resample
## 1 0.9908280 0.9884002    Fold4
## 2 0.9872579 0.9838791    Fold3
## 3 0.9913354 0.9890385    Fold2
## 4 0.9900637 0.9874328    Fold5
## 5 0.9859837 0.9822681    Fold1
```

## Prediction

Finally, we make predictions for "classe" using our GBM model on the 20 cases in the testing dataset. 


```r
# make predictions on test set
gbmPred <- predict.train(gbmFit,testing)
gbmPred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


