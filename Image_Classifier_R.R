---
  title: "Build a Deep Learning Based Image Classifier"
output: html_notebook
---
## Task 1: Import Libraries
#> install.packages("keras")
library(keras)
install_keras()

#Import the Fashion MNIST Dataset which consists of 70k Gray-scale images of 28*28 scale and 10 clothing categories
fashion_mnist <- dataset_fashion_mnist()
library(magrittr)
library(dplyr)
#Partition data into training and test (60k training size/10 test size)
#Create objects for train and test
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test
class_names = c('T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot')

#Data Exploration
dim(train_images)
dim(train_labels)
#Each label is a subset between 0 to 9
train_labels[1:20]
dim(test_images)
dim(test_labels)
library(help = "datasets")

library(tidyr)
library(ggplot2)
#Pre-process dataset
#Currently, gray scale image pixel is denoted from 0 to 255, Lets scale it from 0 to 1 (Normalize) for better training results and best practice

train_images <- train_images / 255
test_images <- test_images / 255
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0),xaxs='i', yaxs='i')
for (i in 1:25) {
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev))
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}
#Building Model
#Using keras model API sequential to build the model
#Flatten layer from 28*28 2D array into 1D array vector (length 784)
model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128,activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')
summary(model)

#Compile Model
#Add loss and optimizer, Minimize the loss
#Sparse categorical crossentropy as integer values of the labels
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

#Train and evaluate the model
model %>% fit(
  train_images, train_labels,
  epochs=10, validation_split=0.2
)
#Test the test data which is unseen with ground truth labels
score <- model %>% evaluate(test_images, test_labels)
cat('Test loss:',score["loss"], "\n")
cat('Test accuracy:', score["accuracy"], "\n")

#Make Predictions on Test Data
predictions <- model %>% predict(test_images)
#See first prediction
predictions[1,]
#Check which label or prediction has the highest confidence
which.max(predictions[1,])

class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]

#Check test label for 1st item
test_labels[1]

img <- test_images[1, , , drop = FALSE]
dim(img)
predictions <- model %>% predict(img)
predictions
# subtract 1 as labels are 0-based
prediction <- predictions[1, ] - 1
which.max(prediction)

class_pred <- model %>% predict_classes(img)
class_pred

#Plot images with predictions
#Red-Incorrect, Green-correct

## Plot Images with Predictions ##

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  predicted_label1 <- which.max(predictions[i,]) - 1
  true_label <- test_labels[i]
  if (predicted_label1 == true_label) {
    color <- '#008800'
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label1 + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}



par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  print(i)
  print(predictions)
#  print(predictions)
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}


