################################################################################
# Data Mining: Exam 2
#
# PCA w/ regression and neural nets
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
################################################################################
# set memory limits
#options(java.parameters = "-Xmx64048m") # 64048 is 64 GB
#memory.limit(size=10000000000024)
################################################################################
# load data
################################################################################
# note that the features (1stFlrSF,2ndFlrSF) will get automatically renamed to 
#("X1stFlrSF","X2ndFlrSF") because names in R cannot begin with numbers

setwd("/Users/prasad/Desktop/Data Mining/Exam")
tr <- read.table("train.csv", header=T, sep=",", quote="",
                colClasses=c("numeric",rep("factor",2),rep("numeric",2),rep("factor",12)
                             ,rep("numeric",4),rep("factor",5),"numeric",rep("factor",7)
                             ,"numeric","factor",rep("numeric",3),rep("factor",4)
                             ,rep("numeric",10),"factor","numeric","factor"
                             ,"numeric",rep("factor",2),"numeric","factor"
                             ,rep("numeric",2),rep("factor",3),rep("numeric",6)
                             ,rep("factor",3),rep("numeric",3),rep("factor",2)
                             ,"numeric")
                )
te <- read.table("test.csv", header=T, sep=",", quote="",
                colClasses=c("numeric",rep("factor",2),rep("numeric",2),rep("factor",12)
                             ,rep("numeric",4),rep("factor",5),"numeric",rep("factor",7)
                             ,"numeric","factor",rep("numeric",3),rep("factor",4)
                             ,rep("numeric",10),"factor","numeric","factor"
                             ,"numeric",rep("factor",2),"numeric","factor"
                             ,rep("numeric",2),rep("factor",3),rep("numeric",6)
                             ,rep("factor",3),rep("numeric",3),rep("factor",2))
                )

################################################################################
# EDA
################################################################################
# Q1) percent of complete records using the DataQualityReportOverall() function
source("DataQualityReportOverall.R")
DataQualityReportOverall(tr)

# Q2) percent of complete records by variable using the DataQualityReport() function
source("DataQualityReport.R")
attr = DataQualityReport(tr)
miss <- attr[attr$NumberMissing!=0,]
miss
################################################################################
# Preprocess data
################################################################################
# Q3) delete the features that have more than 80% of their values missing, as well as the id column
tr<- subset(tr, select = -c(Alley, PoolQC, Fence, MiscFeature, Id))
te<- subset(te, select = -c(Alley, PoolQC, Fence, MiscFeature))

# Q4) for the records with missing values, impute them using the mice package
set.seed(2016)
library(mice)
tri <- mice(tr, m=1, method='cart', printFlag=FALSE)
tei <- mice(te, m=1, method='cart', printFlag=FALSE)

# At this point, you could just save your workspace (Global environment) so your 
# imputed datasets are saved using the save.image() function. You can easily 
# load them back later without re-imputing using the load() function
#setwd("E:\\Dropbox\\_Purdue\\_Teaching\\DM\\_Assignments & Exams\\2_Fall 2017\\Exam2\\housing data")
save.image(file="imputeexam.RData")
load(file="imputeexam.RData")

################################################################################
# Q5) Zero- and Near Zero-Variance Predictors
library(caret)
dim(tri) # dimension of dataset
tri <- complete(tri)
tei <- complete(tei)
dim(tei)
tri_filtered <- subset(tri, select = -c(nearZeroVar(tri,uniqueCut = 3)))
nearZeroVar(tri,uniqueCut = 3)
tri <- tri_filtered
rm(tri_filtered)
# keep features in tei that were kept in tri
tei <- tei[,c("Id",names(tri)[1:(ncol(tri)-1)])]

################################################################################
# Q6) Creating Dummy Variables
library(caret)
# create dummies 
dummies <- dummyVars(SalePrice ~ ., data = tri)
ex <- data.frame(predict(dummies, newdata = tri))
names(ex) <- gsub("\\.", "", names(ex)) # removes dots from col names
tri <- ex

dummies <- dummyVars(~ ., data = tei)
ex <- data.frame(predict(dummies, newdata = tei))
names(ex) <- gsub("\\.", "", names(ex)) # removes dots from col names
tei <- ex
rm(dummies, ex)
# ensure only features available in both train and score sets are kept
tri <- tri[,c(Reduce(intersect, list(names(tri), names(tei))))]
tei <- tei[,c("Id", Reduce(intersect, list(names(tri), names(tei))))]

tri <- cbind(tr$SalePrice, tri)
names(tri)[1] <- "SalePrice"
################################################################################
# Q7) Identify Correlated Predictors and remove them
descrCor <-  cor(tri[,2:ncol(tri)])  # correlation matrix
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .999) # number of features having a correlation greater than some value
summary(descrCor[upper.tri(descrCor)])  # summarize the correlations
# which columns in your correlation matrix have a correlation greater than some
# specified absolute cutoff
highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.90)
filteredDescr <- tri[,2:ncol(tri)][,-highlyCorDescr] # remove those specific columns from your dataset
descrCor2 <- cor(filteredDescr) # calculate a new correlation matrix
# summary those correlations to see if all features are now within our range
summary(descrCor2[upper.tri(descrCor2)])
tri <- cbind(tri$SalePrice, filteredDescr)
names(tri)[1] <- "SalePrice"
rm(filteredDescr, descrCor, descrCor2, highCorr, highlyCorDescr)
# ensure same features show up in scoring set
tei <- tei[,c("Id", Reduce(intersect, list(names(tri), names(tei))))]

################################################################################
# Q8) Identifying Linear Dependencies and remove them
# Find if any linear combinations exist and which column combos they are
comboInfo <- caret::findLinearCombos(tri)
tri <- subset(tri, select=-comboInfo$remove)
## comboInfo returns NULL
rm(comboInfo)
# ensure same features show up in scoring set
tei <- tei[,c("Id", Reduce(intersect, list(names(tri), names(tei))))]

################################################################################
# Q9) The remaining features that are truely categorical features, make sure they are
# defined as factors. Below are all the possibilities that you might get using
# dummyVars()
str(tri)
names(tri)
cols <- c("Id","MSSubClass120","MSSubClass160","MSSubClass180","MSSubClass190",
"MSSubClass20","MSSubClass30","MSSubClass40","MSSubClass45","MSSubClass50",
"MSSubClass60","MSSubClass70","MSSubClass75","MSSubClass80","MSSubClass85",
"MSSubClass90","MSZoningCall","MSZoningFV","MSZoningRH","MSZoningRL","MSZoningRM",
"StreetGrvl","StreetPave","AlleyGrvl","AlleyPave","LotShapeIR1","LotShapeIR2",
"LotShapeIR3","LotShapeReg","LandContourBnk","LandContourHLS","LandContourLow",
"LandContourLvl","UtilitiesAllPub","UtilitiesNoSeWa","LotConfigCorner","LotConfigCulDSac",
"LotConfigFR2","LotConfigFR3","LotConfigInside","LandSlopeGtl","LandSlopeMod",
"LandSlopeSev","NeighborhoodBlmngtn","NeighborhoodBlueste","NeighborhoodBrDale",
"NeighborhoodBrkSide","NeighborhoodClearCr","NeighborhoodCollgCr","NeighborhoodCrawfor",
"NeighborhoodEdwards","NeighborhoodGilbert","NeighborhoodIDOTRR","NeighborhoodMeadowV",
"NeighborhoodMitchel","NeighborhoodNAmes","NeighborhoodNoRidge","NeighborhoodNPkVill",
"NeighborhoodNridgHt","NeighborhoodNWAmes","NeighborhoodOldTown","NeighborhoodSawyer",
"NeighborhoodSawyerW","NeighborhoodSomerst","NeighborhoodStoneBr","NeighborhoodSWISU",
"NeighborhoodTimber","NeighborhoodVeenker","Condition1Artery","Condition1Feedr",
"Condition1Norm","Condition1PosA","Condition1PosN","Condition1RRAe","Condition1RRAn",
"Condition1RRNe","Condition1RRNn","Condition2Artery","Condition2Feedr","Condition2Norm",
"Condition2PosA","Condition2PosN","Condition2RRAe","Condition2RRAn","Condition2RRNn",
"BldgType1Fam","BldgType2fmCon","BldgTypeDuplex","BldgTypeTwnhs","BldgTypeTwnhsE",
"HouseStyle15Fin","HouseStyle15Unf","HouseStyle1Story","HouseStyle25Fin","HouseStyle25Unf",
"HouseStyle2Story","HouseStyleSFoyer","HouseStyleSLvl","RoofStyleFlat","RoofStyleGable",
"RoofStyleGambrel","RoofStyleHip","RoofStyleMansard","RoofStyleShed","RoofMatlClyTile",
"RoofMatlCompShg","RoofMatlMembran","RoofMatlMetal","RoofMatlRoll","RoofMatlTarGrv",
"RoofMatlWdShake","RoofMatlWdShngl","Exterior1stAsbShng","Exterior1stAsphShn",
"Exterior1stBrkComm","Exterior1stBrkFace","Exterior1stCBlock","Exterior1stCemntBd",
"Exterior1stHdBoard","Exterior1stImStucc","Exterior1stMetalSd","Exterior1stPlywood",
"Exterior1stStone","Exterior1stStucco","Exterior1stVinylSd","Exterior1stWdSdng",
"Exterior1stWdShing","Exterior2ndAsbShng","Exterior2ndAsphShn","Exterior2ndBrkCmn",
"Exterior2ndBrkFace","Exterior2ndCBlock","Exterior2ndCmentBd","Exterior2ndHdBoard",
"Exterior2ndImStucc","Exterior2ndMetalSd","Exterior2ndOther","Exterior2ndPlywood",
"Exterior2ndStone","Exterior2ndStucco","Exterior2ndVinylSd","Exterior2ndWdSdng",
"Exterior2ndWdShng","MasVnrTypeBrkCmn","MasVnrTypeBrkFace","MasVnrTypeNone",
"MasVnrTypeStone","ExterQualEx","ExterQualFa","ExterQualGd","ExterQualTA",
"ExterCondEx","ExterCondFa","ExterCondGd","ExterCondPo","ExterCondTA","FoundationBrkTil",
"FoundationCBlock","FoundationPConc","FoundationSlab","FoundationStone","FoundationWood",
"BsmtQualEx","BsmtQualFa","BsmtQualGd","BsmtQualTA","BsmtCondFa","BsmtCondGd",
"BsmtCondPo","BsmtCondTA","BsmtExposureAv","BsmtExposureGd","BsmtExposureMn",
"BsmtExposureNo","BsmtFinType1ALQ","BsmtFinType1BLQ","BsmtFinType1GLQ","BsmtFinType1LwQ",
"BsmtFinType1Rec","BsmtFinType1Unf","BsmtFinType2ALQ","BsmtFinType2BLQ","BsmtFinType2GLQ",
"BsmtFinType2LwQ","BsmtFinType2Rec","BsmtFinType2Unf","HeatingGasA","HeatingGasW",
"HeatingGrav","HeatingOthW","HeatingWall","HeatingQCEx","HeatingQCFa","HeatingQCGd",
"HeatingQCPo","HeatingQCTA","CentralAirN","CentralAirY","ElectricalFuseA",
"ElectricalFuseF","ElectricalFuseP","ElectricalMix","ElectricalSBrkr","KitchenQualEx",
"KitchenQualFa","KitchenQualGd","KitchenQualTA","FunctionalMaj1","FunctionalMaj2",
"FunctionalMin1","FunctionalMin2","FunctionalMod","FunctionalSev","FunctionalTyp",
"FireplaceQuEx","FireplaceQuFa","FireplaceQuGd","FireplaceQuPo","FireplaceQuTA",
"GarageType2Types","GarageTypeAttchd","GarageTypeBasment","GarageTypeBuiltIn",
"GarageTypeCarPort","GarageTypeDetchd","GarageFinishFin","GarageFinishRFn",
"GarageFinishUnf","GarageQualEx","GarageQualFa","GarageQualGd","GarageQualPo",
"GarageQualTA","GarageCondEx","GarageCondFa","GarageCondGd","GarageCondPo",
"GarageCondTA","PavedDriveN","PavedDriveP","PavedDriveY","PoolQCEx","PoolQCFa",
"PoolQCGd","FenceGdPrv","FenceGdWo","FenceMnPrv","FenceMnWw","MiscFeatureGar2",
"MiscFeatureOthr","MiscFeatureShed","MiscFeatureTenC","SaleTypeCOD","SaleTypeCon",
"SaleTypeConLD","SaleTypeConLI","SaleTypeConLw","SaleTypeCWD","SaleTypeNew",
"SaleTypeOth","SaleTypeWD","SaleConditionAbnorml","SaleConditionAdjLand",
"SaleConditionAlloca","SaleConditionFamily","SaleConditionNormal","SaleConditionPartial")
cols <- Reduce(intersect, list(names(tri), cols))
tri[cols] <- lapply(tri[cols], factor)
tei[cols] <- lapply(tei[cols], factor)

################################################################################
# Q10) standardize the input features using the preProcess() using a min-max normalization 
# (aka "range"), in addition to using a"YeoJohnson" transformation to make the 
# features more bell-shaped

preProcValues <- preProcess(tri[,2:ncol(tri)], method = c("range", "YeoJohnson"))
# predict() actually transforms the variables
trit <- predict(preProcValues, tri)


# preprocess the test set the same way and call it "teit"
preProcValues <- preProcess(tri[,2:ncol(tei)], method = c("range", "YeoJohnson"))
teit <- predict(preProcValues, tei)

################################################################################
# Q11) 
dim(trit)
dim(teit)

# Q12) subset your trit dataset by creating a new dataset called trit_num, which
# only contains the numeric features (INCLUDING your target variable)
nums <- sapply(trit, is.numeric)
trit_num <- trit[ ,nums]
# subset your teit dataset by creating a new dataset called teit_num, which
# only contains the numeric features (INCLUDING your target variable)
nums2 <- sapply(teit, is.numeric)
teit_num <- teit[ ,nums2]

################################################################################
# Dimension Reduction - PCA
################################################################################
# Q13) 
# LINES OF NEEDED CODE MISSING HERE - STUDENT NEEDS TO FIGURE OUT WHAT GOES HERE
set.seed(12346)
# select randomly 50% of the dataset to serve as training data
library(caret)
inTrain <- createDataPartition(y = trit_num$LotArea, times =1, p=0.5, list=FALSE)  
# train and test sets having the input features
train <- trit_num[inTrain,]
test <- trit_num[-inTrain,]

# Saving Sale Price for train and test
train_sp <- train$SalePrice
test_sp <- test$SalePrice

train <- subset( train, select = -SalePrice )
test <- subset( test, select = -SalePrice )

# perform PCA using the principal() function from the psych package on the train set
library(psych)
names(train)
pca1 <- principal(train
                  , nfactors = 12     # number of componets to extract
                  , rotate = "varimax"  # can specify different rotations
                  , scores = T       # find component scores or not
)

# Q14) obtain eignvalues
pca1$values

# Q15) How much does the first component account for of the total variance in this dataset?
pca1$loadings
# 18.7%

# Q16) Based on the proportion of variance explained criterion, how many components would
# you keep based on a 65% cutoff?
# 9 components

# Q17) Using the eigenvalue criterion how many components would you keep?
length(pca1[[1]][pca1[[1]] > 1])

# Q18) generate a scree plot and decide how many components to keep?
par(mfrow=c(1,1))
plot(pca1$values, type="b", main="Scree plot"
     , col="blue", xlab="Component #", ylab="Eigenvalues", pch=19)

# Q19) perform validation of pca on your test set using the # of PCs you believe makes
# sense based on what the previous 3 criterion led to you choose. Based on the these
# results ONLY (without profiling) do you believe this PCA is repeatable?
pca2 <- principal(test
                  , nfactors = 8     # number of componets to extract
                  , rotate = "varimax"  # can specify different rotations
                  , scores = T       # find component scores or not
)
pca2$loadings


################################################################################
# Create PCs as input features to be used for prediction
################################################################################
# Q20) Now, run PCA on the entire "trit_num" dataset (which ignores the target), using
# the number of components that are features in the dataset
pca_tr <- principal(trit_num[,2:ncol(trit_num)]
                  , nfactors = 12     # number of componets to extract
                  , rotate = "varimax"  # can specify different rotations
                  , scores = T       # find component scores or not
)
# repeat this step to obtain the pc rotations for the teit_num dataset
pca_te <- principal(teit_num[,2:ncol(teit_num)]
                 , nfactors = 12     # number of componets to extract
                 , rotate = "varimax"  # can specify different rotations
                 , scores = T       # find component scores or not
)

# obtain the prinipcal component scores so we can use those as possible
# input features later. Call the PC scores for the training data "tr_pcscores"
# and the PC scores on the scoring set "te_pcscores"
tr_pcscores <- data.frame(predict(pca_tr, data=trit_num[,2:ncol(trit_num)]))
te_pcscores <- data.frame(predict(pca_te, data=teit_num[,2:ncol(teit_num)]))
# just keep the first x # of columns (PCs) based on your decision in how many
# to retain from previous analyses. Hint: You should not be keeping 29 columns.
tr_pcscores <- tr_pcscores[,1:8]
te_pcscores <- te_pcscores[,1:8]

################################################################################
# Make sure datasets for all experiments are standardized similarly
################################################################################
# We are going to use three different datasets and try various approaches
# to modeling to see what works best for this regression-type problem
# 1) trit: imputed data with min-max and Yeo-Johnson transformations
#    teit is our scoring set        
# 2) tr_pcscores: retained principal components as features
#    te_pcscores is our scoring set

# Q21) On the tr_pcscores and te_pcscores datasets, perform a mix-max normalization 
# (aka range) and YeoJohnson transformation and overwrite (i.e. call them the same 
# names) those datasets with those standardized values.
preProcValues <- preProcess(tr_pcscores[,1:ncol(tr_pcscores)], method = c("range","YeoJohnson"))
tr_pcscores <- predict(preProcValues, tr_pcscores)
preProcValues <- preProcess(te_pcscores[,1:ncol(te_pcscores)], method = c("range","YeoJohnson"))
te_pcscores <- predict(preProcValues, te_pcscores)
colnames(te_pcscores) <- c("RC1","RC2","RC3","RC4","RC9","RC12","RC7","RC10")

# 3) tr_scoresNfactors: retained principal components + factor features
#    tr_scoresNfactors is our scoring set
facs <- sapply(trit, is.factor)
trit_facs <- trit[ ,facs]
tr_scoresNfactors <- data.frame(tr_pcscores, trit_facs)
facs2 <- sapply(teit, is.factor)
teit_facs <- teit[ ,facs2]
te_scoresNfactors <- data.frame(te_pcscores, teit_facs)

# ensure the target variables is in the new datasets
tr_pcscores <- data.frame(trit$SalePrice, tr_pcscores); names(tr_pcscores)[1] <- "SalePrice"
tr_scoresNfactors <- data.frame(trit$SalePrice, tr_scoresNfactors); names(tr_scoresNfactors)[1] <- "SalePrice"

################################################################################
# R Environment cleanup 
################################################################################
# At this point you probably have alot of variables in your R environment. Lets
# clean this up these items we don't need anymore
rm( pca1, pca2, preProcValues, trit_facs
   , trit_num, facs, facs2, DataQualityReport, DataQualityReportOverall, pca_te
   , pca_tr, teit_facs, teit_num, train, test, nums, nums2, miss, attr,inTrain)

################################################################################
# Model building 
################################################################################
# Q22) Make sure you set a seed of 1234 before partitioning your data into train
# and test sets.
set.seed(1234)
library(caret)

# create an 80/20 train and test set for each of the three "tr" datasets (trit
#, tr_pcscores, tr_scoresNfactors) using the createDataPartition() function
trainIndex <- createDataPartition(trit$SalePrice # target variable vector
                                  , p = 0.80    # % of data for training
                                  , times = 1   # Num of partitions to create
                                  , list = F    # should result be a list (T/F)
)
train1 <- trit[trainIndex,]
test1 <- trit[-trainIndex,]

train2 <- tr_pcscores[trainIndex,]
test2 <- tr_pcscores[-trainIndex,]


train3 <- tr_scoresNfactors[trainIndex,]
test3 <- tr_scoresNfactors[-trainIndex,]
################################################################################
# multiple linear regression (forward and backward selection) on trit (train1) dataset
################################################################################
#Q23)
m1f <- lm(SalePrice ~ ., data=train1)
summary(m1f)

# forward selection
library(leaps)
mlf <- regsubsets(SalePrice ~ ., data=train1, nbest=1, intercept=T, method='forward') #plot(mlf)
vars2keep <- data.frame(summary(mlf)$which[which.max(summary(mlf)$adjr2),])
names(vars2keep) <- c("keep")  
head(vars2keep)
library(data.table)
vars2keep <- setDT(vars2keep, keep.rownames=T)[]
vars2keep <- c(vars2keep[which(vars2keep$keep==T & vars2keep$rn!="(Intercept)"),"rn"])[[1]]
vars2keep


m1f <- lm(SalePrice ~ LotArea+NeighborhoodNoRidge+OverallQual+YearRemodAdd+
            BsmtQualEx+BsmtFinSF1+GrLivArea+KitchenQualEx+GarageCars, data=train1)
summary(m1f)

# backward selection
library(leaps)
mlb <- regsubsets(SalePrice ~ ., data=train1, nbest=1, intercept=T, method='backward') #plot(mlf)
vars2keep <- data.frame(summary(mlb)$which[which.max(summary(mlb)$adjr2),])
names(vars2keep) <- c("keep")  
head(vars2keep)
library(data.table)
vars2keep <- setDT(vars2keep, keep.rownames=T)[]
vars2keep <- c(vars2keep[which(vars2keep$keep==T & vars2keep$rn!="(Intercept)"),"rn"])[[1]]
vars2keep

m1b <- lm(SalePrice ~ LotArea+NeighborhoodNoRidge+OverallQual+OverallCond+YearBuilt+KitchenQualEx+
            BsmtQualEx+BsmtExposureGd+GrLivArea, data=train1)
summary(m1b)

# Q24) perform regression diagnostics for forward and backward selection models and 
# discuss if you see any potential issues or assumption violations.
source("myDiag.R")
myDiag(m1f)
myDiag(m1b)

# plot predicted vs actual
par(mfrow=c(1,2))
yhat_m1f <- predict(m1f, newdata=train1); plot(train1$SalePrice, yhat_m1f)
yhat_m1b <- predict(m1b, newdata=train1); plot(train1$SalePrice, yhat_m1b)

################################################################################
# multiple linear regression on tr_pcscores (train2) dataset
################################################################################
# Q25)
m2 <- lm(SalePrice~., data=train2)
summary(m2)

# plot predicted vs actual
myDiag(m2)
par(mfrow=c(1,1))
yhat_m2 <- predict(m2, newdata=train2); plot(train2$SalePrice, yhat_m2)

################################################################################
# multiple linear regression on tr_scoresNfactors (train3) dataset
################################################################################
# Q26)
fctr <- lapply(train3[sapply(train3, is.factor)], droplevels)
sapply(fctr, nlevels)
#train3 <- subset(train3, select=-HeatingQCPo)
m3 <- lm(SalePrice ~., data=train3)
summary(m3)
# plot predicted vs actual
par(mfrow=c(1,1))
yhat_m3 <- predict(m3, newdata=train3); plot(train3$SalePrice, yhat_m3)

# Q27)
library(leaps)
mlb3 <- regsubsets(SalePrice ~ ., data=train3, nbest=1, intercept=T, method='backward') #plot(mlf)
vars2keep <- data.frame(summary(mlb3)$which[which.max(summary(mlb)$adjr2),])
names(vars2keep) <- c("keep")  
head(vars2keep)
library(data.table)
vars2keep <- setDT(vars2keep, keep.rownames=T)[]
vars2keep <- c(vars2keep[which(vars2keep$keep==T & vars2keep$rn!="(Intercept)"),"rn"])[[1]]
vars2keep

modelFormula <- paste("SalePrice ~ RC1+RC2+RC3+RC12+RC7+NeighborhoodNoRidge+BsmtExposureGd+BsmtQualEx+KitchenQualEx") 
m3b <- lm(modelFormula, data=train3)
summary(m3b)

# perform regression diagnostics and discuss if you see any potential issues or 
# assumption violations.
source("myDiag.R")
myDiag(m3b)

# plot predicted vs actual
par(mfrow=c(1,2))
yhat_m3 <- predict(m3, newdata=train3); plot(train3$SalePrice, yhat_m3)
yhat_m3b <- predict(m3b, newdata=train3); plot(train3$SalePrice, yhat_m3b)

################################################################################
# Neural Networks
################################################################################
# Q28)
ctrl<- trainControl(method="cv", number=5, savePredictions = TRUE)

# Q29) HINT: I got 12 features
(maxvalue <- summary(trit$SalePrice)["Max."][[1]])

nnet1 <- train(SalePrice/755000 ~ LotArea + OverallQual + OverallCond + YearBuilt + BsmtExposureGd + YearRemodAdd + BsmtQualEx + 
                                KitchenQualEx + GrLivArea + NeighborhoodNoRidge + BsmtFinSF1 + GarageCars,
                  data = train1,     # training set used to build model
                  method = "nnet",     # type of model you want to build
                  trControl = ctrl,    # how you want to learn
                  tuneLength = 15,
                  maxit = 100,
                  metric = "RMSE"     # performance measure
)

# Q30) This code shows your "best" tuning parameters for this neural network.
nnet1$finalModel$tuneValue

# Q31) Now define a custom grid of tuning parameters that can be fed into the tuneGrid
# argument. Try 5 values for the number of hidden nodes, where 2 below and 2 above
# the best value you found using tuneLengh. Likewise, try +/- 0.01 of your "best"
# decay found using tuneLength including the best value. Do not use a decay lower than 0.
# Also, use a maxit = 500.
myGrid <-  expand.grid(size = c(3,4,5,7,9)     # number of units in the hidden layer.
                       , decay = c(0
                                   ,0.00128976776
                                   ,0.00028976776
                                   ,0.01023896776))  #parameter for weight decay. 
nnet1b <- train(SalePrice/755000 ~ LotArea + OverallQual + OverallCond + YearBuilt + BsmtExposureGd + YearRemodAdd + BsmtQualEx + 
                  KitchenQualEx + GrLivArea + NeighborhoodNoRidge + BsmtFinSF1 + GarageCars,
               data = train1,     # training set used to build model 
               method = "nnet",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               tuneGrid = myGrid,
               maxit = 500,
               metric = "RMSE"     # performance measure
)

par(mfrow=c(1,2))
yhat_nn1 <- predict(nnet1, newdata=train1)*maxvalue; plot(train1$SalePrice, yhat_nn1)
yhat_nn1b <- predict(nnet1b, newdata=train1)*maxvalue; plot(train1$SalePrice, yhat_nn1b)

# Q32)  Did your tuning parameter values change?
nnet1b$finalModel$tuneValue

# Q33) Neural network on train2
(maxvalue <- summary(tr_pcscores$SalePrice)["Max."][[1]])

nnet2 <- train(SalePrice/755000 ~ .,
               data = train2,     # training set used to build model
               method = "nnet",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               tuneLength = 15,
               maxit = 100,
               metric = "RMSE"     # performance measure
)
nnet2$finalModel$tuneValue
# Q34) Now define a custom grid of tuning parameters that can be fed into the tuneGrid
# argument. Try 5 values for the number of hidden nodes, where 2 below and 2 above
# the best value you found using tuneLengh. Likewise, try +/- 0.01 of your "best"
# decay found using tuneLength including the best value. Do not use a decay lower than 0.
# Also, use a maxit = 500.

myGrid <-  expand.grid(size = c(8,9,10,11,12)     # number of units in the hidden layer.
                       , decay = c(0
                                   ,0.000701704
                                   ,0.007017038
                                   ,0.017017038))  #parameter for weight decay. 
nnet2b <- train(SalePrice/755000 ~ .,
                data = train2,     # training set used to build model 
                method = "nnet",     # type of model you want to build
                trControl = ctrl,    # how you want to learn
                tuneGrid = myGrid,
                maxit = 500,
                metric = "RMSE"     # performance measure
)


par(mfrow=c(1,2))
yhat_nn2 <- predict(nnet2, newdata=train2)*maxvalue; plot(train2$SalePrice, yhat_nn2)
yhat_nn2b <- predict(nnet2b, newdata=train2)*maxvalue; plot(train2$SalePrice, yhat_nn2b)


# Q35) Neural network on train3
nnet3 <- train(SalePrice/755000 ~ RC1+RC2+RC3+RC12+RC7+NeighborhoodNoRidge+BsmtExposureGd+BsmtQualEx+KitchenQualEx,
               data = train3,     # training set used to build model
               method = "nnet",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               tuneLength = 15,
               maxit = 100,
               metric = "RMSE"     # performance measure
)

nnet3$finalModel$tuneValue

# Q36) Now define a custom grid of tuning parameters that can be fed into the tuneGrid
# argument. Try 5 values for the number of hidden nodes, where 2 below and 2 above
# the best value you found using tuneLengh. Likewise, try +/- 0.01 of your "best"
# decay found using tuneLength including the best value. Do not use a decay lower than 0.
# Also, use a maxit = 500.

myGrid <-  expand.grid(size = c(3,4,5,6,7)     # number of units in the hidden layer.
                       , decay = c(0
                                   ,0.000242447
                                   ,0.002424462
                                   ,0.012424462))  #parameter for weight decay. 
nnet3b <- train(SalePrice/755000 ~ RC1+RC2+RC3+RC12+RC7+NeighborhoodNoRidge+BsmtExposureGd+BsmtQualEx+KitchenQualEx,
                data = train3,     # training set used to build model 
                method = "nnet",     # type of model you want to build
                trControl = ctrl,    # how you want to learn
                tuneGrid = myGrid,
                maxit = 500,
                metric = "RMSE"     # performance measure
)

par(mfrow=c(1,2))
yhat_nn3 <- predict(nnet3, newdata=train3)*maxvalue; plot(train3$SalePrice, yhat_nn3)
yhat_nn3b <- predict(nnet3b, newdata=train3)*maxvalue; plot(train3$SalePrice, yhat_nn3b)
nnet3b$finalModel$tuneValue

################################################################################
# Decision Trees
################################################################################
# Q37)
library(tree)
treefull1 = tree(SalePrice ~ .
               , control = tree.control(nobs=nrow(train1)[[1]]
                                        , mincut = 0
                                        , minsize = 1
                                        , mindev = 0.01)
               , data = train1)
summary(treefull1)
plot(treefull1); text(treefull1, pretty=0) # plot the tree

# perform cross-validation to find optimal number of terminal nodes
cv.treefull1 = cv.tree(treefull1)
par(mfrow=c(1,1))
plot(cv.treefull1$size
     , cv.treefull1$dev
     , type = 'b')

# prune tree where the number of terminal nodes is 6
tree1 = prune.tree(treefull1,best=6)
summary(tree1)
plot(tree1); text(tree1, pretty=0)

# Q38)
yhat_tree1 <- predict(tree1, newdata=train1); plot(train1$SalePrice, yhat_tree1)

# Q39)
## bagged tree on train1
tree1b <- train(SalePrice ~ .,
               data = train1,     # training set used to build model
               method = "treebag",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               metric = "RMSE"     # performance measure
)

## bagged tree on train2
tree2 <- train(SalePrice ~ .,
               data = train2,     # training set used to build model
               method = "treebag",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               metric = "RMSE"     # performance measure
)

## bagged tree on train3
tree3 <- train(SalePrice ~ .,
               data = train3,     # training set used to build model
               method = "treebag",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               metric = "RMSE"     # performance measure
)

par(mfrow=c(2,2))
yhat_dt1 <- predict(tree1, newdata=train1); plot(train1$SalePrice, yhat_dt1)
yhat_dt1b <- predict(tree1b, newdata=train1); plot(train1$SalePrice, yhat_dt1b)
yhat_dt2 <- predict(tree2, newdata=train2); plot(train2$SalePrice, yhat_dt2)
yhat_dt3 <- predict(tree3, newdata=train3); plot(train3$SalePrice, yhat_dt3)

################################################################################
# Model Evaluation
################################################################################
# Q40)
set.seed(12346)
yhat_m1f_te <- predict(m1f, newdata=test1)
yhat_m1b_te <- predict(m1b, newdata=test1)
yhat_m2_te <- predict(m2, newdata=test2)
yhat_m3_te <- predict(m3, newdata=test3)
yhat_m3b_te <- predict(m3b, newdata=test3)

yhat_nn1_te <- predict(nnet1, newdata=test1)*maxvalue
yhat_nn1b_te <- predict(nnet1b, newdata=test1)*maxvalue
yhat_nn2_te <- predict(nnet2, newdata=test2)*maxvalue
yhat_nn2b_te <- predict(nnet2b, newdata=test2)*maxvalue
yhat_nn3_te <- predict(nnet3, newdata=test3)*maxvalue
yhat_nn3b_te <- predict(nnet3b, newdata=test3)*maxvalue

yhat_dt1_te <- predict(tree1, newdata=test1)
yhat_dt1b_te <- predict(tree1b, newdata=test1)
yhat_dt2_te <- predict(tree2, newdata=test2)
yhat_dt3_te <- predict(tree3, newdata=test3)

####################################### PRASAD'S WORK ####################################################


############################################ xgboost ############################################################

library(xgboost)
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3)

xgb.grid <- expand.grid(nrounds = 500,
                        max_depth = seq(6,10),
                        eta = c(0.01,0.3, 1),
                        gamma = c(0.0, 0.2, 1),
                        colsample_bytree = c(0.5,0.8, 1),
                        min_child_weight=seq(1,10),
                        subsample=0.8
)

xgb_tune <-train(SalePrice ~.,
                 data=train1,
                 method="xgbTree",
                 metric = "RMSE",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid
)

yhat_xgb_tr <- predict(xgb_tune,train1)
plot(train1$SalePrice, yhat_xgb_tr)

yhat_xgb_te <- predict(xgb_tune,test1)
plot(test1$SalePrice, yhat_xgb_te)

y_pred <- predict(xgb_tune,teit)
dfp = data.frame(Id,y_pred)
colnames(dfp) <- c("Id","SalePrice")
write.table(dfp, 'Prasad3.csv', quote=F, sep=",", row.names=F, col.names=T)


#xgb_tune3 <-train(SalePrice ~.,
#                 data=train3,
#                 method="xgbTree",
#                 metric = "RMSE",
#                 trControl=cv.ctrl,
#                 tuneGrid=xgb.grid
#)
#y_pred <- predict(xgb_tune,test3)
#plot(test3$SalePrice, y_pred)
##################################################################################################################


yhat_m1f_tr <- predict(m1f, newdata=train1)
yhat_m1b_tr <- predict(m1b, newdata=train1)
yhat_m2_tr <- predict(m2, newdata=train2)
yhat_m3_tr <- predict(m3, newdata=train3)
yhat_m3b_tr <- predict(m3b, newdata=train3)

yhat_nn1_tr <- predict(nnet1, newdata=train1)*maxvalue
yhat_nn1b_tr <- predict(nnet1b, newdata=train1)*maxvalue
yhat_nn2_tr <- predict(nnet2, newdata=train2)*maxvalue
yhat_nn2b_tr <- predict(nnet2b, newdata=train2)*maxvalue
yhat_nn3_tr <- predict(nnet3, newdata=train3)*maxvalue
yhat_nn3b_tr <- predict(nnet3b, newdata=train3)*maxvalue

yhat_dt1_tr <- predict(tree1, newdata=train1)
yhat_dt1b_tr <- predict(tree1b, newdata=train1)
yhat_dt2_tr <- predict(tree2, newdata=train2)
yhat_dt3_tr <- predict(tree3, newdata=train3)

yhat_xgb_tr <- predict(xgb_tune,train1)
yhat_meta_B1 <- predict(nnet_meta, newdata= truncB1)

B1 = data.frame(yhat_m1f_tr, yhat_m1b_tr, yhat_m2_tr, yhat_m3_tr, yhat_m3b_tr, yhat_nn1_tr, yhat_nn1b_tr, yhat_nn2_tr
                ,yhat_nn2b_tr, yhat_nn3_tr, yhat_nn3b_tr, yhat_dt1_tr, yhat_dt1b_tr, yhat_dt2_tr, yhat_dt3_tr, yhat_xgb_tr, train1$SalePrice)

set.seed(12346)
yhat_m1f_te <- predict(m1f, newdata=test1)
yhat_m1b_te <- predict(m1b, newdata=test1)
yhat_m2_te <- predict(m2, newdata=test2)
yhat_m3_te <- predict(m3, newdata=test3)
yhat_m3b_te <- predict(m3b, newdata=test3)

yhat_nn1_te <- predict(nnet1, newdata=test1)*maxvalue
yhat_nn1b_te <- predict(nnet1b, newdata=test1)*maxvalue
yhat_nn2_te <- predict(nnet2, newdata=test2)*maxvalue
yhat_nn2b_te <- predict(nnet2b, newdata=test2)*maxvalue
yhat_nn3_te <- predict(nnet3, newdata=test3)*maxvalue
yhat_nn3b_te <- predict(nnet3b, newdata=test3)*maxvalue

yhat_dt1_te <- predict(tree1, newdata=test1)
yhat_dt1b_te <- predict(tree1b, newdata=test1)
yhat_dt2_te <- predict(tree2, newdata=test2)
yhat_dt3_te <- predict(tree3, newdata=test3)

yhat_xgb_te <- predict(xgb_tune,test1)
yhat_meta_C1 <- predict(nnet_meta, newdata= truncC1)

C1 = data.frame(yhat_m1f_te, yhat_m1b_te, yhat_m2_te, yhat_m3_te, yhat_m3b_te, yhat_nn1_te, yhat_nn1b_te, yhat_nn2_te
                ,yhat_nn2b_te, yhat_nn3_te, yhat_nn3b_te, yhat_dt1_te, yhat_dt1b_te, yhat_dt2_te, yhat_dt3_te, yhat_xgb_te)
colnames(C1) <- c('yhat_m1f_tr', 'yhat_m1b_tr', 'yhat_m2_tr', 'yhat_m3_tr', 'yhat_m3b_tr', 'yhat_nn1_tr', 'yhat_nn1b_tr', 'yhat_nn2_tr'
                  ,'yhat_nn2b_tr', 'yhat_nn3_tr', 'yhat_nn3b_tr', 'yhat_dt1_tr', 'yhat_dt1b_tr', 'yhat_dt2_tr', 'yhat_dt3_tr','yhat_xgb_tr')

truncC1 <- C1[,c(6,7,16)]
preProcValues <- preProcess(truncC1, method = c("center","scale"))
truncC1 <- predict(preProcValues, truncC1)


library(caret)
library(randomForest)

truncB1 <- B1[,c(6,7,16)]
preProcValues <- preProcess(truncB1, method = c("center","scale"))
truncB1 <- predict(preProcValues, truncB1)
newB1 <- cbind(truncB1,B1$train1.SalePrice)
colnames(newB1)[ncol(newB1)] <- 'SalePrice'

nnet_meta <- train(SalePrice/755000 ~ .,
                   data = newB1,     # training set used to build model
                   method = "nnet",     # type of model you want to build
                   trControl = ctrl,    # how you want to learn
                   tuneLength = 15,
                   maxit = 100,
                   metric = "RMSE"     # performance measure
)
nnet_meta$finalModel$tuneValue

myGrid <-  expand.grid(size = c(21,22,23,24,25)     # number of units in the hidden layer.
                       , decay = c(0
                                   ,0.0000170
                                   ,0.000171254
                                   ,0.00170701))  #parameter for weight decay. 
nnet_meta <- train(SalePrice/755000 ~ .,
                   data = newB1,     # training set used to build model 
                   method = "nnet",     # type of model you want to build
                   trControl = ctrl,    # how you want to learn
                   tuneGrid = myGrid,
                   maxit = 500,
                   metric = "RMSE"     # performance measure
)

yhat_meta_B1 <- predict(nnet_meta, newdata= truncB1)
plot(train1$SalePrice, yhat_meta_B1)

yhat_meta_C1 <- predict(nnet_meta, newdata= truncC1)
plot(test1$SalePrice, yhat_meta_C1)


D1 <- data.frame(yhat_m1f_test, yhat_m1b_test, yhat_m2_test, yhat_m3_test, yhat_m3b_test, yhat_nn1_test, yhat_nn1b_test, yhat_nn2_test
                 ,yhat_nn2b_test, yhat_nn3_test, yhat_nn3b_test, yhat_dt1_test, yhat_dt1b_test, yhat_dt2_test, yhat_dt3_test,y_pred)

colnames(D1) <- c('yhat_m1f_tr', 'yhat_m1b_tr', 'yhat_m2_tr', 'yhat_m3_tr', 'yhat_m3b_tr', 'yhat_nn1_tr', 'yhat_nn1b_tr', 'yhat_nn2_tr'
                  ,'yhat_nn2b_tr', 'yhat_nn3_tr', 'yhat_nn3b_tr', 'yhat_dt1_tr', 'yhat_dt1b_tr', 'yhat_dt2_tr', 'yhat_dt3_tr','yhat_xgb_tr')

truncD1 <- D1[,c(6,7,16)]
preProcValues <- preProcess(truncD1, method = c("center","scale"))
truncD1 <- predict(preProcValues, truncD1)


Id <- teit$Id
teit <- subset(teit, select = -Id)

yhat_m1f_test <- predict(m1f, newdata=teit)
yhat_m1b_test <- predict(m1b, newdata=teit)
yhat_m2_test <- predict(m2, newdata=te_pcscores)
yhat_m3_test <- predict(m3, newdata=te_scoresNfactors)
yhat_m3b_test <- predict(m3b, newdata=te_scoresNfactors)

yhat_nn1_test <- predict(nnet1, newdata=teit)*maxvalue
yhat_nn1b_test <- predict(nnet1b, newdata=teit)*maxvalue
yhat_nn2_test <- predict(nnet2, newdata=te_pcscores)*maxvalue
yhat_nn2b_test <- predict(nnet2b, newdata=te_pcscores)*maxvalue
yhat_nn3_test <- predict(nnet3, newdata=te_scoresNfactors)*maxvalue
yhat_nn3b_test <- predict(nnet3b, newdata=te_scoresNfactors)*maxvalue

yhat_dt1_test <- predict(tree1, newdata=teit)
yhat_dt1b_test <- predict(tree1b, newdata=teit)
yhat_dt2_test <- predict(tree2, newdata=te_pcscores)
yhat_dt3_test <- predict(tree3, newdata=te_scoresNfactors)

yhat_xgb_test <- predict(xgb_tune, newdata=teit)
PredSalePrice<- predict(nnet_meta,truncD1)*755000



PredSalePrice<- predict(nnet_meta,truncD1)*755000
# Write out file to be uploaded to Kaggle.com for scoring
avg = (D1[7]+D1[16])/2.0
dfp = data.frame(Id,avg) 

#dfp = data.frame(Id,PredSalePrice)
colnames(dfp) <- c("Id","SalePrice")
write.table(dfp, 'Prasad5.csv', quote=F, sep=",", row.names=F, col.names=T)


#avg4= (yhat_xgb_te + yhat_nn1_te)/2
plot(test1$SalePrice, avg4)
summary(yhat_nn1_te)
# Prasad1 original with nn1 alone
# Prasad2 was average of nn1 and nn1b
# Prasad3 was xgb alone
# Prasad4 was avg of xgb and nn1b
############################################## EXAM #############################################################



results <- matrix(rbind(
cbind(t(postResample(pred=yhat_m1f, obs=train1$SalePrice)), t(postResample(pred=yhat_m1f_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_m1b, obs=train1$SalePrice)), t(postResample(pred=yhat_m1b_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_m2, obs=train2$SalePrice)), t(postResample(pred=yhat_m2_te, obs=test2$SalePrice))),
cbind(t(postResample(pred=yhat_m3, obs=train3$SalePrice)), t(postResample(pred=yhat_m3_te, obs=test3$SalePrice))),
cbind(t(postResample(pred=yhat_m3b, obs=train3$SalePrice)), t(postResample(pred=yhat_m3b_te, obs=test3$SalePrice))),
  
cbind(t(postResample(pred=yhat_nn1, obs=train1$SalePrice)), t(postResample(pred=yhat_nn1_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_nn1b, obs=train1$SalePrice)), t(postResample(pred=yhat_nn1b_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_nn2, obs=train2$SalePrice)), t(postResample(pred=yhat_nn2_te, obs=test2$SalePrice))),
cbind(t(postResample(pred=yhat_nn2b, obs=train2$SalePrice)), t(postResample(pred=yhat_nn2b_te, obs=test2$SalePrice))),
cbind(t(postResample(pred=yhat_nn3, obs=train3$SalePrice)), t(postResample(pred=yhat_nn3_te, obs=test3$SalePrice))),
cbind(t(postResample(pred=yhat_nn3b, obs=train3$SalePrice)), t(postResample(pred=yhat_nn3b_te, obs=test3$SalePrice))),
    
cbind(t(postResample(pred=yhat_dt1, obs=train1$SalePrice)), t(postResample(pred=yhat_dt1_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_dt1b, obs=train1$SalePrice)), t(postResample(pred=yhat_dt1b_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_dt2, obs=train2$SalePrice)), t(postResample(pred=yhat_dt2_te, obs=test2$SalePrice))),
cbind(t(postResample(pred=yhat_dt3, obs=train3$SalePrice)), t(postResample(pred=yhat_dt3_te, obs=test3$SalePrice))),
cbind(t(postResample(pred=yhat_xgb_tr, obs=train1$SalePrice)), t(postResample(pred=yhat_xgb_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_meta_B1, obs=train1$SalePrice)), t(postResample(pred=yhat_meta_C1, obs=test1$SalePrice)))
), nrow=17)
results<-results[,-c(3,6)]
colnames(results) <- c("Train_RMSE", "Train_R2","Test_RMSE", "Test_R2")
rownames(results) <- c("MLR_Forward","MLR_Backward","MLR_PCs","MLR_PCs+Factors",
                       "MLR_Backward_PCs+Factors","NN_ForBackFeatures","NN_ForBackFeatures_Optimized",
                       "NN_PCs","NN_PCs_Optimized","NN_BackFeatures","NN_BackFeatures_Optimized",
                       "Tree_Numerics+Factors","BaggedTree_Numerics+Factors",
                       "BaggedTree_PCs","BaggedTree_PCs+Factors","XGB","meta")
results

library(reshape)
results <- melt(results)
names(results) <- c("Model","Stat","Values")

# Q41 and 42)
library(ggplot2)
# RMSE
p1 <- ggplot(data=results[which(results$Stat=="Train_RMSE" | results$Stat=="Test_RMSE"),]
            , aes(x=Model, y=Values, fill=Stat)) 
p1 <- p1 + geom_bar(stat="identity", color="black", position=position_dodge()) + theme_minimal()
p1 <- p1 + facet_grid(~Model, scale='free_x', drop = TRUE)
p1 <- p1 + scale_fill_manual(values=c('#FF6666','blue'))
p1 <- p1 + xlab(NULL) + theme(axis.text.x = element_text(angle = 90, hjust=1, vjust=.5))

p1 <- p1 + theme(strip.text.x = element_text(size=0, angle=0, colour="white"),
               strip.text.y = element_text(size=0, face="bold"),
               strip.background = element_rect(colour="white", fill="white"))
p1 <- p1 + ggtitle("RMSE Performance")
p1

# R2
p2 <- ggplot(data=results[which(results$Stat=="Train_R2" | results$Stat=="Test_R2"),]
             , aes(x=Model, y=Values, fill=Stat)) 
p2 <- p2 + geom_bar(stat="identity", color="black", position=position_dodge()) + theme_minimal()
p2 <- p2 + facet_grid(~Model, scale='free_x', drop = TRUE)
p2 <- p2 + scale_fill_manual(values=c('#FF6666','blue'))
p2 <- p2 + xlab(NULL) + theme(axis.text.x = element_text(angle = 90, hjust=1, vjust=.5))

p2 <- p2 + theme(strip.text.x = element_text(size=0, angle=0, colour="white"),
                 strip.text.y = element_text(size=0, face="bold"),
                 strip.background = element_rect(colour="white", fill="white"))
p2 <- p2 + ggtitle("R2 Performance")
p2


################################################################################
# Score data / Deployment
################################################################################
# Q43)
Id <- teit$Id
teit <- subset(teit, select = -Id)

PredSalePrice<- predict(nnet1,teit[c('LotArea','OverallQual','OverallCond','YearBuilt','BsmtExposureGd'
                                   ,'YearRemodAdd','BsmtQualEx','KitchenQualEx','GrLivArea'
                                   ,'NeighborhoodNoRidge','BsmtFinSF1','GarageCars')])
PredSalePrice <- PredSalePrice * 755000


# Write out file to be uploaded to Kaggle.com for scoring

dfp = data.frame(Id,PredSalePrice) 
colnames(dfp) <- c("Id","SalePrice")
write.table(dfp, 'Prasad1.csv', quote=F, sep=",", row.names=F, col.names=T)
# Q44) Upload predictions to Kaggle to be scored

