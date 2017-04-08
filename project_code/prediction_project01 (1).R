#####  GLM3 interactions ######
set.seed(123)
Scaled_project01_No2016 <- as.data.frame(scale(project01_No2016[,-c(1,2,4,8,10,11)]))
Scaled_project01_No2016$Sex <- project01_No2016$Sex
Scaled_project01_No2016$ParticipationIn2016 <- project01_No2016$ParticipationIn2016
Scaled_project01_No2016 <- Scaled_project01_No2016[sample(nrow(Scaled_project01_No2016)),]
glm_fit_T <- glm(ParticipationIn2016 ~ . + Year.Category.:ParticipationBefore2016 , Scaled_project01_No2016[,-4], family = binomial)
summary(glm_fit_T)
vif(glm_fit3)

#### Prediction GLM3 ####
#put Max age for each participant
summerized_project01 <- project01

summerized_project01$Age.Category <- ave(project01$Age.Category , project01$Id, FUN=max)
summerized_project01$Year <- ave(project01$Year , project01$Id, FUN=max)
summerized_project01$Year.Category. <- ave(project01$Year.Category. , project01$Id, FUN=max)
summerized_project01$Rank <- ave(project01$Rank , project01$Id)
summerized_project01$Time <- ave(project01$Time , project01$Id)
summerized_project01$Pace <- ave(project01$Pace , project01$Id)

summerized_project01 <- summerized_project01[!duplicated(summerized_project01),]


Scaled_project01 <- as.data.frame(scale(summerized_project01[,-c(1,2,4,8,10,11)]))
Scaled_project01$ID <- summerized_project01$Id
Scaled_project01$Sex <- summerized_project01$Sex
Scaled_project01$ParticipationIn2016 <- summerized_project01$ParticipationIn2016
#Scaled_project01 <- Scaled_project01[sample(nrow(Scaled_project01)),]
glm_prob_T <- predict.glm(glm_fit_T, newdata = Scaled_project01, type = 'response')
glm_pred_T <- ifelse(glm_prob_T>0.5,1,0)
error_rate3 <- mean(glm_pred_T!=Scaled_project01$ParticipationIn2016)
error_rate3
glm_pred_T <- as.data.frame(as.matrix(glm_pred_T))
glm_pred_T$Id <- Scaled_project01$ID

##### GD predictions #########

###### probability prediction GD #####
# now that we have the optimal values for weights we can test them on the test data and see what is the
# error rate
Scaled_project01_GD <- cbind(rep(1,nrow(Scaled_project01)), Scaled_project01) 
pred_GD_T <- sigmoid(as.matrix(Scaled_project01_GD[,-c(8,10)])%*%theta_GD)
pred_GD_T <- ifelse(pred_GD_T>.5,1,0)
pred_GD_T <- as.data.frame(pred_GD_T)
pred_GD_T$ID <- Scaled_project01_GD$ID
Error_GD_T <- mean(pred_GD_T!=Scaled_project01_GD$ParticipationIn2016)
Error_GD_T 






write.csv(glm_pred_T, file = "/Users/haniehkashani/Documents/Machine learning mcgill/project01/glm.prediction.csv",
          row.names=FALSE)

write.csv(pred_GD_T, file = "/Users/haniehkashani/Documents/Machine learning mcgill/project01/GradientD.prediction.csv",
          row.names=FALSE)
