rm(list=ls(all=T))
setwd("F:/data Scientist/project 1 bike renting/R code")
# load libraries
x= c("ggplot2","corrgram","DataCombine","scales","psych","gplots","Metrics" ,"inTrees","DMwR","car", "caret","rlang","usdm", "Information", "randomForest", "unbalanced", "C50", "dummies", "e1071", "MASS", "ROSE", "rpart", "gbm")
install.packages(x)
lapply(x, require, character.only=TRUE)
rm(x)
day = read.csv("day.csv", header=T )
#load data
strings = c(" ", "", "NA")
day_copy=day
str(day)

#### EXPLORATORY DATA ANALYSIS

# convert variables to proper data type
day$season=as.factor(day$season)
day$yr=as.factor(day$yr)
day$mnth=as.factor(day$mnth)
day$holiday=as.factor(day$holiday)
day$weekday=as.factor(day$weekday)
day$workingday=as.factor(day$workingday)
day$weathersit=as.factor(day$weathersit)
day$dteday=as.Date(day$dteday)

###### Visualizations related to given data

# histograms of numerical variables
hist(day$temp)
hist(day$atemp)
hist(day$hum)
hist(day$windspeed)
hist(day$cnt)

# let us plot bar graph of each catagorical variable w.r.t target variable
# bike rent count w.r.t. month
ggplot(day, aes_string(x=day$mnth, y=day$cnt))+ 
  geom_bar(stat='identity', fill="DarkSlateBlue")+ theme_bw() + 
  xlab("month") +ylab("count")+
  ggtitle("Bike rent count w.r.t. month")+
  theme(text=element_text(size=15 ))

#bike rent count w.r.t. weekday
ggplot(day, aes_string(x=day$weekday, y=day$cnt))+ 
  geom_bar(stat='identity', fill="DarkSlateBlue")+ theme_bw() + 
  xlab("weekday") +ylab("count")+
  ggtitle("bike rent count w.r.t. weekday")+
  theme(text=element_text(size=15 ))

#bike rent count w.r.t. season
ggplot(day, aes_string(x=day$season, y=day$cnt))+ 
  geom_bar(stat='identity', fill="DarkSlateBlue")+ theme_bw() + 
  xlab("Season") +ylab("count")+
  ggtitle("bike rent count w.r.t. season")+
  theme(text=element_text(size=15 ))

#bike rent count w.r.t. year
ggplot(day, aes_string(x=day$yr, y=day$cnt))+ 
  geom_bar(stat='identity', fill="DarkSlateBlue")+ theme_bw() + 
  xlab("year") +ylab("count")+
  ggtitle("bike rent count w.r.t. year")+
  theme(text=element_text(size=15 ))

# scatter plot analysis

# scatter plot of hum vs cnt and weathersit
ggplot(day, aes_string(x=day$hum, y=day$cnt))+geom_point(aes_string(colour= day$weathersit), size=4)+
  theme_bw()+ xlab("hum")+ylab("bike rent count")+ggtitle("scatter plot analysis hum vs cnt")+ theme(text=element_text(size=15))+
  scale_color_discrete(name="weathersit")

# scatter plot windspeed vs cnt and weathersit,season
ggplot(day, aes_string(x=day$windspeed, y=day$cnt))+geom_point(aes_string(colour= day$weathersit, shape=day$season), size=4)+
  theme_bw()+ xlab("windspeed")+ylab("Bike rent count")+ggtitle("scatter plot windspeed vs cnt")+ theme(text=element_text(size=15))+
  scale_color_discrete(name="weathersit")+ scale_shape_discrete(name= "season")

# scatter plot of temp vs cnt and weathersit
ggplot(day, aes_string(x=day$temp, y=day$cnt))+geom_point(aes_string(colour= day$weathersit), size=4)+
  theme_bw()+ xlab("temp")+ylab("bike rent count")+ggtitle("scatter plot analysis temp vs cnt")+ theme(text=element_text(size=15))+
  scale_color_discrete(name="weathersit")

# scatter plot of atemp vs cnt and season
ggplot(day, aes_string(x=day$atemp, y=day$cnt))+geom_point(aes_string(colour= day$season), size=4)+
  theme_bw()+ xlab("atemp")+ylab("bike rent count")+ggtitle("scatter plot analysis a atemp vs cnt")+ theme(text=element_text(size=15))+
  scale_color_discrete(name="season")

###### DATA PREPROCESSING


#finding missing values in whole data
sum(is.na(day))

# No Missing values in data

####### outlier analysis 

cnames_for_outlier= colnames(day[,c('temp', 'atemp', 'hum', 'windspeed')])


# find the outliers using box plot
for (i in 1: length(cnames_for_outlier)){
  assign(paste0("ab", i), ggplot(aes_string(y=(cnames_for_outlier[i])), 
  data=subset(day))+stat_boxplot(geom = 'errorbar', width=0.5)+ 
    geom_boxplot(outlier.colour = "red", 
  fill= "grey", outlier.shape = 18, 
  outlier.size = 1, notch = FALSE)+ 
    theme(legend.position = "bottom")+ 
    labs(y=cnames_for_outlier[i])+ggtitle(paste("box plot of cnt for", cnames_for_outlier[i])))
}


#plotting plots together
gridExtra::grid.arrange(ab1,ab2, ncol=2)
gridExtra::grid.arrange(ab3,ab4, ncol=2)



#replace outliers with na
for(i in cnames_for_outlier){
  val=day[,i][day[,i] %in% boxplot.stats(day[,i])$out]
  day[,i][day[,i] %in% val] = NA
}

sum(is.na(day))
#impute outliers
day$windspeed[is.na(day$windspeed)]= mean(day$windspeed,na.rm= T)
day$hum[is.na(day$hum)]= mean(day$hum, na.rm=T)

sum(is.na(day))

for (i in 1: length(cnames_for_outlier)){
  assign(paste0("cd", i), ggplot(aes_string(y=(cnames_for_outlier[i])), 
                                 data=subset(day))+stat_boxplot(geom = 'errorbar', width=0.5)+ 
           geom_boxplot(outlier.colour = "red", 
                        fill= "grey", outlier.shape = 18, 
                        outlier.size = 1, notch = FALSE)+ 
           theme(legend.position = "bottom")+ 
           labs(y=cnames_for_outlier[i])+ggtitle(paste("box plot of cnt for", cnames_for_outlier[i])))
}

gridExtra::grid.arrange(cd1,cd2, ncol=2)
gridExtra::grid.arrange(cd3,cd4, ncol=2)

# boxplot after outlier analysis
hist(day$hum)
hist(day$windspeed)

#select numeric data for correlation plot
numeric_index=sapply(day[,1:16], is.numeric)
numeric_data= day[,numeric_index]

# Correlation plot to check correlation of independant numeric variables
corrgram(day[,numeric_index],order=F, upper.panel=panel.pie, 
         text.panel=panel.txt, main="correlation plot")


# anova test to check dependancy of catagorical variables with target variable

factor_index=sapply(day, is.factor)
factor_data=day[,factor_index]

for(i in 1:7){
  print(names(factor_data)[i])
  cnt= day[,16]
  anova=aov(cnt~factor_data[,i], data=factor_data)
         print(summary(anova))
}

# Remove unwanted columns from data

day_deleted= subset(day, select=-c(instant, dteday, casual, registered, atemp, holiday, weekday, workingday))
day_deleted

#clean the environment
library(DataCombine)
rmExcept("day","day_deleted" )

########  lets Build the Machine Learning models on the data 
#data is having numeric variable as a target variable
# so we will build following Regression models
# Decision Tree, Linear Regression, Random Forest model and choose one with higher accuracy


# divide data into train and test using Simple Random sampling method

train_index=sample(1:nrow(day_deleted), 0.8*nrow(day_deleted),replace=FALSE , prob = NULL)
train=day_deleted[train_index,]
test=day_deleted[-train_index,]
######### DECISION TREE MODEL OF REGRESSION

#Decision tree for regression
fit=rpart(cnt~., data= train, method="anova")

# predict for new test data
predictions_dt= predict(fit, test[,-8])

# compare real and predicted values of target variable

comparision_dt=data.frame("Real"=test[,8], "Predicted"= predictions_dt)

# function to calculate MAPE
mape= function(y, yhat){
  mean(abs((y-yhat)/y))*100
}
mape(test[,8], predictions_dt)
 
# To find rmse and mae
rmse(test[,8],predictions_dt)
mae(test[,8],predictions_dt)

# compare real and predicted values of target variable

comparison_dt=data.frame("Real"=test[,8], "Predicted"= predictions_dt)

# plotting the graph for real and predicted values of target variable
plot(test$cnt, type="l", lty=4, col="violet")
lines(predictions_dt, col="red")

##Predictive performance of model using error metrics 

# mape(error rate)= 18.59%
# Accuracy =81.41%
# rmse=875.8407
# mae=652.9456 



###### LINEAR REGRESSION MODEL 

#check multicolinearity
vif(day_deleted[,5:7])

# run regression model
lr_model=lm(cnt~., data=train)

#summary of regression model
summary(lr_model)

# prediction on test data
predictions_lr=predict(lr_model, test[,1:7])

# calculate mape,rmse,mae

mape= function(y, yhat){
  mean(abs((y-yhat)/y))*100
}
# calculate error metrics to evaluate the model

mape(test[,8],predictions_lr)
rmse(test[,8],predictions_lr)
mae(test[,8],predictions_lr)

# plotting the graph for real and predicted values of target variable

plot(test$cnt, type="l", lty=4, col="violet")
lines(predictions_lr, col="red")

#example of output with sample input of 2nd row

predict(lr_model, test[2,])

# compare real and predicted values of target variable

comparison_lr=data.frame("Real"=test[,8], "Predicted"= predictions_lr)


# #Predictive performance of model using error metrics 

# mape(error rate)= 15.43%
# Accuracy of Linear regression model= 84.57%
# adjusted R-squared= 85.22% which is greater than 80%
# rmse=756.5901
# mae= 569.2133

##### RANDOM FOREST MODEL OF REGRESSION

# creating the model 
RF_model= randomForest(cnt~., train, importance=TRUE, ntree= 120)

# convert rf object to trees
treeList= RF2List(RF_model)         

# extract the rules from the model
rules=extractRules(treeList, train[,-8])
rules[1:2,]

# make rule readable

readablerules=presentRules(rules,colnames(train))

# get rule metrics 

ruleMetric=getRuleMetric(rules, train[,-8], train$cnt)

# predict the test data using RF model
predictions_rf=predict(RF_model,test[,-8])

# Calculate error metrics to evaluate the performance of model

mape(test[,8],predictions_rf)
rmse(test[,8],predictions_rf)
mae(test[,8],predictions_rf)


# plotting the graph for real and predicted values of target variable

plot(test$cnt, type="l", lty=4, col="violet")
lines(predictions_rf, col="red")

# compare real and predicted values of target variable

comparison_rf=data.frame("Real"=test[,8], "Predicted"= predictions_rf)
#example of output with sample input of 2nd row

test[2,]
predict(RF_model, test[2,])

test[87,]
predict(RF_model, test[87,])

# #Predictive performance of model using error metrics 

# mape(error rate)= 13.24%
# Accuracy of Random Forest Model= 86.76%
# rmse=650.57
# mae= 484.68

####################################################################################
# So after comparing the performances of all three models 
# we can see that the Random Forest model is having high accuracy  
# so we will select Random Forest Model to predict the bike rental count.

# selecting the same model as output model of this project
RF_model= randomForest(cnt~., train, importance=TRUE, ntree= 120)

# predict test data
predictions_rf=predict(RF_model, test[,-8])

# To plot the error w.r.t no of tress use for predicting output
plot(RF_model)

# show the important variables by using plot
varImpPlot(RF_model)

# plotting the graph for real and predicted values of target variable in test data

plot(test$cnt, predictions_rf, xlab= 'Real_values', ylab= 'predicted_values', main='Rf_model for test data')

# calculate error metrics MAPE to see the result of model
mape(test$cnt,predictions_rf)

##### Now saving the output of RF model 
# creating the new variable "predicted" in the given data
day_deleted$predicted= with(day, predict(RF_model, day[,-8]))
day$predicted= day_deleted$predicted
rmExcept(c("day", "RF_model", "mape" , "day_deleted"))
write.csv(day, 'R_output_main.csv', row.names=F)
write.csv(day[c("season","yr","mnth","weathersit","temp","hum","windspeed","cnt","predicted")], 'R_output.csv',row.names=F)







