# Code run to output .RData objects to read into markup. These were variables
# that could not be computed directly when knitting the rmd document.

setwd(".")
require(caret); require(randomForest)

# Read data
dat = read.csv("data/pml-training.csv")
dat_testing = read.csv("data/pml-testing.csv")
dim(dat)

# Preprocess predictors in training dataset
set.seed(32323)
new_dat <- dat[,-c(1)]   # Remove index variable
new_dat <- new_dat[,-nearZeroVar(new_dat)]   # Remove near-zero-variance features
NAs <- apply(new_dat, 2, function(x) {sum(is.na(x))})   # Remove features with NAs

# Apply preprocessing in training dataset to test dataset
NAt <- table(colnames(new_dat), NAs)
training_final <- new_dat[, which (NAs < 1)]
c <- colnames(training_final)
c <- c[-length(c)]
testing_final <- dat_testing[,c]

#5 Trees : Train and save model
set.seed(32323)
rf_model_5tree <- train(classe ~ ., data=training_final, method="rf", ntree=5,
                trControl=trainControl(method="cv", number=5), prox=TRUE, allowParallel=TRUE)
rf_model_5tree_results <- rf_model_5tree$results
rf_model_5tree_err <- rf_model_5tree$finalModel$err.rate
rf_model_5tree_conf <- rf_model_5tree$finalModel$confusion
save(rf_model_5tree_results, file="data/rf_model_5tree_results.RData")
save(rf_model_5tree_err, file="data/rf_model_5tree_err.RData")
save(rf_model_5tree_conf, file="data/rf_model_5tree_conf.RData")

# 10 Trees : Train and save model
set.seed(32323)
rf_model_10tree <- train(classe ~ ., data=training_final, method="rf", ntree=10,
                        trControl=trainControl(method="cv", number=5), prox=TRUE, allowParallel=TRUE)
rf_model_10tree_results <- rf_model_10tree$results
rf_model_10tree_err <- rf_model_10tree$finalModel$err.rate
rf_model_10tree_conf <- rf_model_10tree$finalModel$confusion
save(rf_model_10tree_results, file="data/rf_model_10tree_results.RData")
save(rf_model_10tree_err, file="data/rf_model_10tree_err.RData")
save(rf_model_10tree_conf, file="data/rf_model_10tree_conf.RData")

#15 Trees : train and save model.
set.seed(32323)
rf_model_15tree <- train(classe ~ ., data=training_final, method="rf", ntree=15,
                        trControl=trainControl(method="cv", number=5), prox=TRUE, allowParallel=TRUE)
rf_model_15tree_results <- rf_model_15tree$results
rf_model_15tree_err <- rf_model_15tree$finalModel$err.rate
rf_model_15tree_conf <- rf_model_15tree$finalModel$confusion
save(rf_model_15tree_results, file="data/rf_model_15tree_results.RData")
save(rf_model_15tree_err, file="data/rf_model_15tree_err.RData")
save(rf_model_15tree_conf, file="data/rf_model_15tree_conf.RData")