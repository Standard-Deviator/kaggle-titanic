# adapted from
# https://blogs.rstudio.com/ai/posts/2021-12-09-keras-preprocessing-layers/
library(tidyverse)
library(tensorflow)
library(tfdatasets)
library(keras)
#library(reticulate)

# import data
data <- read_csv("resources/tweets_combined.csv")

# split
train <- data %>%
  dplyr::sample_frac(0.80)

test  <- dplyr::anti_join(data, train, by = 'ID')

text <- as_tensor(c(
  "From each according to his ability, to each according to his needs!",
  "Act that you use humanity, whether in your own person or in the person of any other, always at the same time as an end, never merely as a means.",
  "Reason is, and ought only to be the slave of the passions, and can never pretend to any other office than to serve and obey them."
))

# Create and adapt layer
text_vectorizer <- layer_text_vectorization(output_mode="int")
text_vectorizer %>% adapt(text)

# Create a simple classification model
input <- layer_input(shape(NULL), dtype="int64")

output <- input %>%
  layer_embedding(input_dim = text_vectorizer$vocabulary_size(),
                  output_dim = 16) %>%
  layer_gru(8) %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(input, output)

# Create a labeled dataset (which includes unknown tokens)
train_dataset <- tensor_slices_dataset(list(train$tweet,
                                            train$target))

# Preprocess the string inputs
train_dataset <- train_dataset %>%
  dataset_batch(2) %>%
  dataset_map(~list(text_vectorizer(.x), .y),
              num_parallel_calls = tf$data$AUTOTUNE)

# Train the model
model %>%
  compile(optimizer = "adam", loss = "binary_crossentropy",metrics = list(c("accuracy","mse"))) %>%
  fit(train_dataset)

# export inference model that accepts strings as input
input <- layer_input(shape = 1, dtype="string")
output <- input %>%
  text_vectorizer() %>%
  model()

end_to_end_model <- keras_model(input, output)

# Test inference model
test_data <- as_tensor(test$tweet)

test_output <- end_to_end_model(test_data)
as.array(test_output)

