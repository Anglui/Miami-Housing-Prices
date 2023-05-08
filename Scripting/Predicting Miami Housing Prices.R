### Setting Up Workspace -------------------------------------------------------

library(boot)
library(caret)
library(leaps)
library(tree)
library(randomForest)
library(gbm)
library(MASS)
library(ggplot2)
library(dplyr)
library(ISLR)
library(reshape2)
library(glmnet)

# Manually lock seed for reproducibility
set.seed(1)

# Load data
housingData = read.csv('miami-housing.csv')
attach(housingData)

# Check for missing values
sum(is.na(housingData))

### Exploratory Data Analaysis -------------------------------------------------

# Review summary statistics
summary(housingData)

# Review correlations
correlationMatrix <- round(cor(housingData),2)

# Get upper triangle of the correlation matrix
get_upper_tri <- function(correlationMatrix){
  correlationMatrix[lower.tri(correlationMatrix)]<- NA
  return(correlationMatrix)
}

upper_tri <- get_upper_tri(correlationMatrix)
upper_tri

# Reorder the correlation matrix
reorder_correlationMatrix <- function(correlationMatrix){
  # Use correlation between variables as distance
  dd <- as.dist((1-correlationMatrix)/2)
  hc <- hclust(dd)
  correlationMatrix <-correlationMatrix[hc$order, hc$order]
}

correlationMatrix <- reorder_correlationMatrix(correlationMatrix)
upper_tri <- get_upper_tri(correlationMatrix)
# Melt the correlation matrix
melted_correlationMatrix <- melt(upper_tri, na.rm = TRUE)

# Create a heatmap
correlationHeatMap <- ggplot(melted_correlationMatrix, aes(Var2, Var1, fill = value)) +
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 60, vjust = 1, 
                                   size = 12, hjust = 1)) +
  coord_fixed()

# Print the heatmap
print(correlationHeatMap)

fullCorrelationData = as.data.frame(correlationMatrix)
fullCorrelationData

# Histgram of Variables
ggplot(data = housingData, aes(x = SALE_PRC)) +
  geom_histogram(bins = 10)

ggplot(data = housingData, aes(x = TOT_LVG_AREA)) +
  geom_histogram(bins = 10)

ggplot(data = housingData, aes(x = LND_SQFOOT)) +
  geom_histogram(bins = 10)

ggplot(data = housingData, aes(x = SPEC_FEAT_VAL)) +
  geom_histogram(bins = 10)

ggplot(data = housingData, aes(x = RAIL_DIST)) +
  geom_histogram(bins = 10)

ggplot(data = housingData, aes(x = CNTR_DIST)) +
  geom_histogram(bins = 10)

ggplot(data = housingData, aes(x = SUBCNTR_DI)) +
  geom_histogram(bins = 10)

ggplot(data = housingData, aes(x = HWY_DIST)) +
  geom_histogram(bins = 10)

ggplot(data = housingData, aes(x = age)) +
  geom_histogram(bins = 10)

ggplot(data = housingData, aes(x = month_sold)) +
  geom_histogram(bins = 10)

ggplot(data = housingData, aes(x = avno60plus)) +
  geom_histogram(bins = 10)

ggplot(data = housingData, aes(x = structure_quality)) +
  geom_histogram(bins = 10)

# Look at scatter plots
ggplot(data = housingData, aes(x = TOT_LVG_AREA , y = SALE_PRC)) + 
  geom_point(alpha = 0.5) +
  geom_smooth()

ggplot(data = housingData, aes(x = month_sold , y = SALE_PRC)) + 
  geom_point(alpha = 0.5) +
  geom_smooth()

### Training Test Holdout (80/20 split) ----------------------------------------
sample = sample(c(TRUE, FALSE), nrow(housingData), replace=TRUE, prob=c(0.8,0.2))
train = housingData[sample,]
test = housingData[!sample,]
dim(train)
dim(test)

### Linear Regression Models ---------------------------------------------------

## Create linear regression model with sale price as response
lm.all = lm(SALE_PRC~. -PARCELNO, data=train) # Remove ParcelNo from regressors
summary(lm.all) # review variables that are statistically significant

# Check R^2
lm.allr2 = summary(lm.all)$adj.r.squared
lm.allr2

## Create linear regression model without non-significant variables
lm.sig = lm(SALE_PRC~. -PARCELNO -WATER_DIST -month_sold, data=train)
summary(lm.sig)

#check R^2
lm.sigr2 = summary(lm.sig)$adj.r.squared
lm.sigr2

#check RMSE for both models
lm.allrmse = sqrt(mean((test$SALE_PRC-predict.glm(lm.all,test))^2))
lm.sigrmse = sqrt(mean((test$SALE_PRC-predict.glm(lm.sig,test))^2))

lm.allrmse
lm.sigrmse
# model 1 has lower RMSE


# Perform 10-fold cross validatino on model
lm.all = glm(SALE_PRC~. -PARCELNO, data=train)
cv.lm.all = cv.glm(train,lm.all, K=10)

#check RMSE of model
summary(cv.lm.all)
cv.delta.all = cv.lm.all$delta
cv.lm.allrmse = sqrt(sum(cv.delta.all)/2)
cv.lm.allrmse

## Next we will create a LASSO model to see if we can get better metrics
xtrain = subset(train, select = -c(PARCELNO, SALE_PRC))
ytrain = subset(train, select = c(SALE_PRC))

xtest = subset(test, select = -c(PARCELNO, SALE_PRC))
ytest = subset(test, select = c(SALE_PRC))

ytrain = as.numeric(unlist(ytrain))
xtrain = data.matrix(xtrain, rownames.force = NA)

ytest = as.numeric(unlist(ytest))
xtest = data.matrix(xtest, rownames.force = NA)

class(ytrain)
class(xtrain)

# Build LASSO model.
cv.out = cv.glmnet(xtrain, ytrain, alpha=1)

par(mfrow = c(1,1))
plot(cv.out) # visualize lasso progress

bestlam = cv.out$lambda.min # select best model
bestlam

lassomodel = glmnet(xtrain,ytrain,alpha=1,lambda=bestlam)

lasso.coef = predict(lassomodel,type="coefficients",s=bestlam)[1:16,]
lasso.coef # best lambda uses all coefficients.

# Calculate test RMSE for LASSO model
lasso.pred = predict(cv.out,s=bestlam ,newx=xtest)
lassoRMSE = sqrt(mean((lasso.pred - ytest)^2))

lm.allrmse
lassoRMSE
# LASSO model has slightly lower RMSE that standard MLR. 
# Check R^2 of LASSO model
lassomodel$dev.ratio

### Ensemble Models ------------------------------------------------------------

## Random Forest model

# Random forest model should set number of variables at split (m)
# number of predictors divided by 3 (p/3). Since model has 15 predictors, m = 5
m = 5

rf.data=randomForest(SALE_PRC~. -PARCELNO, data = train, ntrees = 500, mtry = m, 
                     importance = T)
rf.data

# Hyper parameterization of "m" in random forest 
bestmtry <- tuneRF(xtrain, ytrain, data = train, stepFactor = 1.5, improve = 1e-5, ntree = 500)
print(bestmtry)
# confirms best m = 5.

# Check R^2 for random forest model.
rf.datar2 = mean(rf.data$rsq)
rf.datar2

# Performance on test set
yhat.rf = predict(rf.data, newdata = test)

plot1 = plot(yhat.rf, ytest) # Visualizes random forest error
abline(0, 1)
plot1

# RF test RMSE
rftestrmse = sqrt(mean((yhat.rf - ytest)^2))
rftestrmse

# Determine which variables have greatest importance according to RF
plot(rf.data)
varImpPlot(rf.data) 
# total living area is most important. 

# Build using 10-fold CV.

rf.data.cv = rfcv(xtrain, ytrain, cv.fold = 10)
summary(rf.data.cv)

# Look at metrics.
rf.data.cv$n.var
rf.data.cv$error.cv
plot(rf.data.cv$n.var, rf.data.cv$error.cv)
# look at test RMSE
rfCVTestRMSE = sqrt(rf.data.cv$error.cv[1])
rfCVTestRMSE

## Boosting Model

# Compare boosting to see if results are similar for the importance of variables and test RMSE.
boost.data = gbm(SALE_PRC~. -PARCELNO, data = train, distribution= "gaussian", 
                 cv.folds = 10, n.trees = 5000, interaction.depth = 4)
summary(boost.data)
# once again, total living area is most important variable in sale price.

# Performance on test set
yhat.boost = predict(boost.data, newdata = test)
plot1 = plot(yhat.boost, ytest) # visualize boosting error
abline(0,1) 
plot1

# Test RMSE
boostTestRMSE = sqrt(mean((yhat.boost - ytest)^2))
boostTestRMSE

# Calculate R^2
y_test_mean = mean(ytest)
tss = sum((ytest - y_test_mean)^2)
rss = sum((yhat.boost - ytest)^2)
boostR2 = 1 - (rss/tss)
boostR2


## Build Bagging Model
# Bag model has all predictors at split
bag.data = randomForest(SALE_PRC~. -PARCELNO, data = train, mtry = 15, 
                        importance = T)
bag.data

# Performance on test set
yhat.bag = predict(bag.data, newdata = test)
plot1 = plot(yhat.bag, ytest) # visualize bagging error
abline(0,1)
plot1

# Test MSE
bagTestRMSE = sqrt(mean((yhat.bag-ytest)^2))
bagTestRMSE

# Calculate R^2
mean(bag.data$rsq)

# Variable Importance Plot
plot(bag.data)
varImpPlot(bag.data)

# Compare RMSE across all models
RMSElist = list(lm.allrmse, lm.sigrmse, lassoRMSE, 
                rfCVTestRMSE, boostTestRMSE, bagTestRMSE)

which.min(RMSElist) # check which model had the lowest RMSE