library(dplyr)
library(tidyr)
library(class)
library(caret)
library(e1071)
library(stringr)

df <- read.csv(file.choose(), header = T)
df2 <- read.csv(file.choose(), header = T)


beers <- df
breweries <- df2

breweries
str(breweries)
str(beers)

#count number of breweries by state
breweries %>% group_by(State) %>% summarise(count = n_distinct(Brew_ID))


mergeddf <- merge(beers, breweries, by.x=c("Brewery_id"),
                  by.y=c("Brew_ID"))

range(mergeddf$ABV, na.rm= T)
median(mergeddf$ABV, na.rm=T)

mergeddf[rowSums(is.na(mergeddf)) == 0,]



str(mergeddf)

#columns with na values
colnames(mergeddf)[colSums(is.na(mergeddf)) > 0]

#Compare accuracy on knn by group
#run knn on all groups
#stratify the groups by beer type and run knn 
#compare the accuracies and test if they are significantly different
#Get a sample of different accuracies by using k-fold cross-validation
#Do this by using an anova for means, then test multiple comparisons


colnames(mergeddf)[colSums(is.na(mergeddf)) > 0]


qqnorm(y = mergeddf$ABV)
qqline(y = mergeddf$ABV)

qqnorm(y = mergeddf$IBU)
qqline(y = mergeddf$IBU)

#both data sets are fairly normal so I will impute the columns with the mean

mergeddf$ABV <- sapply(mergeddf$ABV, function(x) ifelse(is.na(x), median(mergeddf$ABV, na.rm = TRUE), x))
mergeddf$IBU <- sapply(mergeddf$IBU, function(x) ifelse(is.na(x), median(mergeddf$IBU, na.rm = TRUE), x))

range(mergeddf$ABV)

#create new factor column that contains 5 levels of main beer classes
#lager, ale, ipa, pilsner, malt, stout, other


#levels(mergeddf$bClass) <- c("Ale", "Lager", "IPA", "Pilsner", "Malt", "Stout", "Other")
mergeddf$bClass <- factor(rep("Other", nrow(mergeddf)))
levels(mergeddf$bClass) <- c("Other","Ale", "IPA")
mergeddf$bClass[which(grepl("[Aa]le", mergeddf$Style) == 1)] <- "Ale"
mergeddf$bClass[which(grepl("IPA", mergeddf$Style) == 1)] <- "IPA"
mergeddf %>% group_by(bClass) %>% summarise(row_count = length(bClass))

#mergeddf$bClass[which(grepl("Lager", mergeddf$Style) == 1)] <- "Lager"
#mergeddf$bClass[which(grepl("Pilsner", mergeddf$Style) == 1)] <- "Pilsner"
#mergeddf$bClass[which(grepl("Malt", mergeddf$Style) == 1)] <- "Malt"
#mergeddf$bClass[which(grepl("Stout", mergeddf$Style) == 1)] <- "Stout"


#find average accuracy on 100 cross validations. for 20 k and get the highest accuracy
#the accuracy of prediction is less using the whole dataset as opposed to using knn on each level
#showing that there are differences in ABV and IBU by beer type

#we will use an 80-20 split
set.seed(100)
#function finds maximum accuracy 
predictKNN <- function(dataframe = df) {
  n = 1
  accuracydf <- data.frame(accuracy = numeric(1000), k = numeric(1000))
  for(i in 1:100) {
    train_indices <- sample(1:nrow(df), round(.70 * nrow(df)))
    train <- df[train_indices,]
    test <- df[-train_indices,]
    
    #storing accuracy data in data frame
    for(j in 1:100) {
      classifications <- knn(train[c("ABV", "IBU")], test[c("ABV", "IBU")], cl = train$bClass, k = j, prob = F)
      CM = confusionMatrix(table(classifications, test$bClass))
      accuracydf$accuracy[n] = CM$overall[1]
      accuracydf$k[n] = j
      n = n + 1
    }
  }
  return(max(accuracydf$accuracy, na.rm = TRUE))
}


accuracydf %>% ggplot(mapping = aes(x = k, y = accuracy)) + geom_point() + geom_smooth() + ggtitle("K vs. Accuracy")

#find the k with highest mean accuracy, you will use that subset of k's to test for significance


#predictive accuracy for each distinct IPA Style when only using "IPA" bClass data vs predictive accuracy for 
#each distinct IPA Style when using C("IPA", "Ale") data
#Do the same for Lagers

#Calculate sensitivity metric for each IPA Style

#Only Use IPA data in the test for IPA round
#Only use Lager Data in the test for Lager round

#Is our predictive accuracy greater than 50% when testing ipa only data against ipa, lager data

ipa_ale_df <- mergeddf %>% filter(bClass %in% c("IPA", "Ale"))
ipa_df <- mergeddf %>% filter(bClass %in% c("IPA"))
ale_df <- mergeddf %>% filter(bClass %in% c("Ale"))

ipa_ale_df$bClass <-  droplevels(ipa_ale_df$bClass)

ipa_ale_df$bClass <- factor(ipa_ale_df$bClass, levels = c("IPA", "Ale"))
levels(ipa_ale_df$bClass)

#removing ale,other levels
ipa_df$bClass <- droplevels(ipa_df$bClass)
#adding ale level to ipa_df
ipa_df$bClass <- factor(ipa_df$bClass, levels = c(levels(ipa_df$bClass), "Ale"))

#same process to ale_df
ale_df$bClass <- droplevels(ale_df$bClass)
#adding ale level to ipa_df
ale_df$bClass <- factor(ale_df$bClass, levels = c(levels(ale_df$bClass), "IPA"))
levels(ale_df$bClass)

nrow(ipa_df)
nrow(ipa_ale_train)
#IPA,ALE VS IPA,ALE (50/50 split)
set.seed(100)
n = 1
accuracydf <- data.frame(accuracy = numeric(10000), k = numeric(10000), 
                         sensitivity = numeric(10000), specificity = numeric(10000))
for(i in 1:100) {
  #Get dataset with evenly distributed IPAs and Ales (50/50)
  ipa_test_ind <- sample(1:nrow(ipa_df), nrow(ipa_df)-250)
  ale_df_ind <- sample(1:nrow(ale_df), 250)
  #building train set
  ipa_ale_train <- rbind(ipa_df[-ipa_test_ind,], ale_df[ale_df_ind,])
  ipa_ale_overall_test <- ipa_ale_df[sapply(ipa_ale_df$Name.x, function(x) x %in% ipa_ale_train$Name.x) == FALSE,]
  ipa_ale_test_ind <- sample(1:nrow(ipa_ale_overall_test), round(.2 * nrow(ipa_ale_train)))
  ipa_ale_test <- ipa_ale_overall_test[ipa_ale_test_ind,]
  
  
  #storing accuracy data in data frame
  for(j in 1:100) {
    classifications <- knn(ipa_ale_train[c("ABV", "IBU")], ipa_ale_test[c("ABV", "IBU")], cl = ipa_ale_train$bClass, k = j, prob = F)
    CM = confusionMatrix(table(classifications, ipa_ale_test$bClass))
    accuracydf$accuracy[n] = CM$overall[1]
    accuracydf$sensitivity[n] = CM$byClass["Sensitivity"]
    accuracydf$specificity[n] = CM$byClass["Specificity"]
    accuracydf$k[n] = j
    n = n + 1
  }
}

max(accuracydf$accuracy, na.rm = TRUE)

accuracydf[accuracydf$accuracy == 0.91,]

summary_acc_df <- accuracydf %>% group_by(k) %>% summarise(mean_accuracy = mean(accuracy), 
                                                           mean_sensitivity = mean(sensitivity),
                                                           mean_specificity = mean(specificity, na.rm = T))
overall_mean_accuracy <- summary_acc_df[which.max(summary_acc_df$mean_accuracy),]
overall_mean_accuracy

acc_df <- as.data.frame(summary_acc_df)
accuracydf %>% ggplot(aes(x=k, y = accuracy)) + geom_point() + geom_smooth()

str(acc_df)
library(ggthemes)
acc_df %>% ggplot(aes(x = k, y = mean_accuracy)) + geom_point() + geom_smooth() + theme_economist() + 
  ggtitle("Mean Accuracy For 100 Iterations vs. K") + xlab("K") + ylab("Mean Accuracy")
#run t-test two test if our overall_mean_accuracy is greater than 50


?t.test

t.test(ipa_ale_df$ABV[ipa_ale_df$bClass=="IPA",], ipa_ale_df)

plot(density(ipa_ale_df$ABV[ipa_ale_df$bClass=="IPA"]))

hist(ipa_ale_df$ABV[ipa_ale_df$bClass=="IPA"], col=rgb(1,0,0,0.5), xlim = range(.01,.15), main="Ale/IPA ABV Histogram")
hist(ipa_ale_df$ABV[ipa_ale_df$bClass=="Ale"], col=rgb(0,0,1,0.5), add=T)

?hist
