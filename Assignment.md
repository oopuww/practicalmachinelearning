## Practical Machine Learning Project
**Predict the manner people did the exercise**

Date: March 31, 2019

### Overview

The objective of this project is to use the data collected through
wearable devices, such as Jawbone Up, Fitbit to monitor personal
activities and predict how they perform the exercise.

The data we use in this analysis is from the source:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>.
We use the training data to create our model and test data to evaluate
the model performance.

### Data Exploration

First, we read in the data and explore the dataset to see the
attributes, data type. We also take a look at the distributiont of our
response variable (classe).

*Please see appendix for outputs*

    library(mlbench)
    library(caret)
    library(parallel)
    library(iterators)
    library(foreach)
    library(doParallel)
    library(corrplot)

    training = read.csv('pml-training.csv',header=T)
    testing = read.csv('pml-testing.csv',header=T)
    head(training)
    dim(training) # 19622 160
    str(training)

    table(training$classe)

    ## 
    ##    A    B    C    D    E 
    ## 5580 3797 3422 3216 3607

### Data Pre-Processing

After viewing the data, we decided to convert factor variables to
numeric variables and check missing data. We tranformed variables with
missing values into indicators.

    training$kurtosis_roll_belt = as.numeric(as.character(training$kurtosis_roll_belt))
    training$kurtosis_picth_belt = as.numeric(as.character(training$kurtosis_picth_belt))
    training$skewness_roll_belt = as.numeric(as.character(training$skewness_roll_belt))
    training$skewness_roll_belt.1 = as.numeric(as.character(training$skewness_roll_belt.1))
    training$max_yaw_belt = as.numeric(as.character(training$max_yaw_belt))
    training$min_yaw_belt = as.numeric(as.character(training$min_yaw_belt))
    training$kurtosis_roll_arm = as.numeric(as.character(training$kurtosis_roll_arm))
    training$kurtosis_picth_arm = as.numeric(as.character(training$kurtosis_picth_arm))
    training$skewness_yaw_arm = as.numeric(as.character(training$skewness_yaw_arm))
    training$kurtosis_roll_dumbbell = as.numeric(as.character(training$kurtosis_roll_dumbbell))
    training$kurtosis_picth_dumbbell = as.numeric(as.character(training$kurtosis_picth_dumbbell))
    training$max_yaw_dumbbell = as.numeric(as.character(training$max_yaw_dumbbell))
    training$min_yaw_dumbbell = as.numeric(as.character(training$min_yaw_dumbbell))
    training$kurtosis_roll_forearm = as.numeric(as.character(training$kurtosis_roll_forearm))
    training$kurtosis_picth_forearm = as.numeric(as.character(training$kurtosis_picth_forearm))
    training$max_yaw_forearm = as.numeric(as.character(training$max_yaw_forearm))
    training$min_yaw_forearm = as.numeric(as.character(training$min_yaw_forearm))

    missing_cnt = sapply(training,function(x) sum(is.na(x)))
    table(missing_cnt) 

    ## missing_cnt
    ##     0 19216 19218 19221 19225 19226 19227 19248 19294 19296 19300 19301 
    ##    76    67     1     3     1     3     1     2     1     1     3     1

    for(col in names(training)){
      if(sum(is.na(training[,col]))>0)
        training[,col] = ifelse(is.na(training[,col]),0,1)
    }

### Model Building

We split the data into training and testing set using 70/30 split.
Excluded 4 variables that are not associated with our response variable.
Then we calculated the correlation of all the variables and keep the top
30 variables that are relatively highly correlated with our response
variable.

    set.seed(123)
    idx = createDataPartition(training$classe,p=0.7,list=FALSE)
    train = training[idx,]
    test = training[-idx,]

    yvar = as.numeric(train$classe)
    xvar = train[,sapply(train,is.numeric)]
    xvar = xvar[,-c(1,2,3,4)]

    allvar = cbind(xvar,yvar)
    m = abs(cor(allvar))
    corr_df = data.frame(row=rownames(m)[row(m)[upper.tri(m)]], 
                         col=colnames(m)[col(m)[upper.tri(m)]], 
                         corr=m[upper.tri(m)])
    corr_df_y = corr_df[which(corr_df$col=='yvar'),]
    corr_df_y2 = corr_df_y[order(-corr_df_y$corr),]

    # take top 30 variables that are most correlated with response

    corr_df_y2[1:30,c(1)]

    ##  [1] pitch_forearm        magnet_arm_x         magnet_belt_y       
    ##  [4] magnet_arm_y         accel_arm_x          accel_forearm_x     
    ##  [7] magnet_forearm_x     magnet_belt_z        pitch_arm           
    ## [10] magnet_arm_z         total_accel_forearm  magnet_dumbbell_z   
    ## [13] total_accel_arm      accel_dumbbell_x     magnet_forearm_y    
    ## [16] accel_arm_y          accel_belt_z         roll_arm            
    ## [19] total_accel_belt     pitch_dumbbell       accel_dumbbell_z    
    ## [22] roll_belt            magnet_dumbbell_x    total_accel_dumbbell
    ## [25] yaw_forearm          roll_dumbbell        yaw_arm             
    ## [28] accel_arm_z          magnet_forearm_z     gyros_dumbbell_y    
    ## 136 Levels: accel_arm_x accel_arm_y accel_arm_z ... yaw_forearm

    corr_org = cor(allvar)
    corrplot(corr_org[corr_df_y2[1:30,c(1)],corr_df_y2[1:30,c(1)]])



#### Random Forest

Use the top 30 variables we selected and create a random foreset model
using 5-fold cross-validation. Then we apply the model to the testing
set. The accurary for the testing set is 0.9867 so the **out-of-sample
error is 0.0133**. We consider this as a decent model so we decided to
use the model in our validation (test) set to predict the 20 new cases.

    cluster = makeCluster(detectCores() - 1)
    registerDoParallel(cluster)
    fitControl = trainControl(method = "cv", number = 5, allowParallel = TRUE)
    rf_model = train(classe ~ pitch_forearm+magnet_arm_x+magnet_belt_y+magnet_arm_y+accel_arm_x+
                     accel_forearm_x+magnet_forearm_x+magnet_belt_z+pitch_arm+magnet_arm_z+
                     total_accel_forearm+magnet_dumbbell_z+total_accel_arm+accel_dumbbell_x+magnet_forearm_y+
                     accel_arm_y+accel_belt_z+roll_arm+total_accel_belt+pitch_dumbbell+
                     accel_dumbbell_z+roll_belt+magnet_dumbbell_x+total_accel_dumbbell+yaw_forearm+
                     roll_dumbbell+yaw_arm+accel_arm_z+magnet_forearm_z+gyros_dumbbell_y, 
                     data=train, method="rf", trControl = fitControl)
    print(rf_model)

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    30 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10989, 10991, 10989, 10990, 10989 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9818732  0.9770669
    ##   16    0.9849310  0.9809383
    ##   30    0.9793256  0.9738426
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 16.

    plot(rf_model,main="Accuracy of Random forest model by number of predictors")



    test_pred = predict(rf_model,newdata=test)
    confusionMatrix(test$classe,test_pred)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1668    3    0    0    3
    ##          B   11 1122    6    0    0
    ##          C    0   19 1002    5    0
    ##          D    0    0   26  936    2
    ##          E    0    1    0    2 1079
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9867          
    ##                  95% CI : (0.9835, 0.9895)
    ##     No Information Rate : 0.2853          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9832          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9934   0.9799   0.9691   0.9926   0.9954
    ## Specificity            0.9986   0.9964   0.9951   0.9943   0.9994
    ## Pos Pred Value         0.9964   0.9851   0.9766   0.9710   0.9972
    ## Neg Pred Value         0.9974   0.9952   0.9934   0.9986   0.9990
    ## Prevalence             0.2853   0.1946   0.1757   0.1602   0.1842
    ## Detection Rate         0.2834   0.1907   0.1703   0.1590   0.1833
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9960   0.9882   0.9821   0.9935   0.9974


