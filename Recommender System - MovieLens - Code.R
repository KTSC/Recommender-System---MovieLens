##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################
rm(list=ls())
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
# set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


requiredPackages <- c("ggplot2", "knitr", "h2o", "data.table", "lubridate")
lapply(requiredPackages, library, character.only = TRUE)


##########################################################
# Make a copy of the original train and test datasets.  
##########################################################

edx_copy <- edx
validation_copy <- validation

##########################################################
# Separate the genres. These lines replicates each ratings for every genre on the movie. 
##########################################################

edx_copy$genres <- lapply(edx_copy$genres, as.character)
validation_copy$genres <- lapply(validation_copy$genres, as.character)
edx_copy <- edx_copy %>% 
  mutate(genre = as.character(genres)) %>%
  separate_rows(genre, sep = "\\|") %>%
  select(-"genres", -"timestamp", -"title")

# Extract the genre in validation datasets
validation_copy <- validation_copy %>% 
  mutate(genre = as.character(genres)) %>%
  separate_rows(genre, sep = "\\|")%>%
  select(-"genres", -"timestamp", -"title")


# First, define the RMSE function:
RMSE <- function(true_ratings = NULL, predicted_ratings = NULL) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# MODEL 1: 
# Most basic assumption. Just go for the average of all movies. 
mu_hat <- mean(edx$rating)

# Predict the RMSE on the validation set
rmse_mean_model_result <- RMSE(validation$rating, mu_hat)

# Creating a results dataframe that contains all RMSE results
results <- data.frame(model="Mean-Baseline Model", RMSE=rmse_mean_model_result)



# Model 2: 
# Just the average, but with replication by genre. 
mu_hat_c <- mean(edx_copy$rating)

# Predict the RMSE on the validation set
rmse_mean_model_result_c <- RMSE(validation_copy$rating, mu_hat_c)

results <- results %>% add_row(model="Mean-Baseline Model with Weighed Genre", RMSE=rmse_mean_model_result_c)


# Model 3: User, movie and genre with regularization
# try new lambdas for the Regularized Movie+User+Genre Based Model. 

lambdas_mug <- seq(3, 7, 0.4)

# Compute the predicted ratings on validation dataset using different values of lambda

rmses <- sapply(lambdas_mug, function(lambda) {
  
  # Calculate the item effect
  b_i <- edx %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu_hat) / (n() + lambda))
  
  # Calculate user effect
  b_u <- edx %>%
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu_hat) / (n() + lambda))
  
  # Calculate the genre effect
  b_u_g <- edx %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genres) %>%
    summarise(b_u_g = sum(rating - b_i - mu_hat - b_u) / (n() + lambda))
  
  # Compute the predicted ratings on validation dataset
  predicted_ratings <- validation %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_u_g, by='genres') %>%
    mutate(pred = mu_hat + b_i + b_u + b_u_g) %>%
    pull(pred)
  
  # Predict the RMSE on the validation set
  return(RMSE(validation$rating, predicted_ratings))
})

# check on the range of lambdas that produces the lowest RMSEs. It is about 15
df_mug <- data.frame(RMSE = rmses, lambdas = lambdas_mug)

# Get the lambda value that minimize the RMSE
min_lambda_mug <- lambdas_mug[which.min(rmses)]

# Predict the RMSE on the validation set
rmse_regularized_movie_user_genre_model <- min(rmses)

# Adding the results to the results dataset
results <- results %>% add_row(model="Regularized Movie+User+Genre Based Model", RMSE=rmse_regularized_movie_user_genre_model)



# Model 4: User, movie and genre with regularization and added weight to movies based on genre. 
# lambdas for the Regularized Movie+User+Genre Based Model. 

lambdas_mug_c <- seq(12, 18, 0.4)

# Compute the predicted ratings on validation dataset using different values of lambda

rmses_c <- sapply(lambdas_mug_c, function(lambda) {
  
  # Calculate the item effect
  b_i <- edx_copy %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu_hat_c) / (n() + lambda))
  
  # Calculate user effect
  b_u <- edx_copy %>%
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu_hat_c) / (n() + lambda))
  
  # Calculate the genre effect
  b_u_g <- edx_copy %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genre) %>%
    summarise(b_u_g = sum(rating - b_i - mu_hat_c - b_u) / (n() + lambda))
  
  # Compute the predicted ratings on validation dataset
  predicted_ratings <- validation_copy %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_u_g, by='genre') %>%
    mutate(pred = mu_hat_c + b_i + b_u + b_u_g) %>%
    pull(pred)
  
  # Predict the RMSE on the validation set
  return(RMSE(validation_copy$rating, predicted_ratings))
})

# check on the range of lambdas that produces the lowest RMSEs. It is about 15
df_mug_c <- data.frame(RMSE = rmses_c, lambdas = lambdas_mug_c)

# Get the lambda value that minimize the RMSE
min_lambda_mug_c <- lambdas_mug_c[which.min(rmses_c)]

# Predict the RMSE on the validation set
rmse_regularized_movie_user_genre_model_c <- min(rmses_c)

# Adding the results to the results dataset
results <- results %>% add_row(model="Regularized Movie+User+Weighed Genre Based Model", RMSE=rmse_regularized_movie_user_genre_model_c)


##########################################################
# Neural Network: Data transformation
##########################################################

# Mutate the timestamp to be 0 or 1 depending on the moment ratings start to have 0.5 granularity = 1045526400
edx <- edx %>% mutate(timestamp_binary = ifelse(edx$timestamp > 1045526400, 1, 0))
validation <- validation %>% mutate(timestamp_binary = ifelse(validation$timestamp > 1045526400, 1, 0))

############
# One-hot encoding of genres
############

genres <- as.data.frame(edx$genres, stringsAsFactors=FALSE)
genres_v <- as.data.frame(validation$genres, stringsAsFactors=FALSE)
# n_distinct(edx_copy$genres)
genres2 <- as.data.frame(tstrsplit(genres[,1], '[|]',
                                   type.convert=TRUE),
                         stringsAsFactors=FALSE)
genres2_v <- as.data.frame(tstrsplit(genres_v[,1], '[|]',
                                     type.convert=TRUE),
                           stringsAsFactors=FALSE)


genre_list <- c("Action", "Adventure", "Animation", "Children",
                "Comedy", "Crime","Documentary", "Drama", "Fantasy",
                "Film-Noir", "Horror", "Imax", "Musical", "Mystery","Romance",
                "Sci-Fi", "Thriller", "War", "Western") # There are 19 genres in total

genre_matrix <- matrix(0, length(edx$movieId)+1, n_distinct(genre_list))                       
genre_matrix[1,] <- genre_list #set first row to genre list

genre_matrix_v <- matrix(0, length(validation$movieId)+1, n_distinct(genre_list))                       
genre_matrix_v[1,] <- genre_list #set first row to genre list

colnames(genre_matrix) <- genre_list #set column names to genre list
colnames(genre_matrix_v) <- genre_list #set column names to genre list

#iterate through matrix
for (i in 1:nrow(genres2)) {
  for (c in 1:ncol(genres2)) {
    genmat_col <- which(genre_matrix[1,] == genres2[i,c])
    genre_matrix[i+1,genmat_col] <- 1L
  }
}

for (i in 1:nrow(genres2_v)) {
  for (c in 1:ncol(genres2_v)) {
    genmat_col <- which(genre_matrix_v[1,] == genres2_v[i,c])
    genre_matrix_v[i+1,genmat_col] <- 1L
  }
}
#convert into dataframe
genre_matrix <- as.data.frame(genre_matrix[-1,], stringsAsFactors=FALSE) #remove first row, which was the genre list
genre_matrix_v <- as.data.frame(genre_matrix_v[-1,], stringsAsFactors=FALSE)

edx_by_gen <- cbind(edx[,1:3], genre_matrix, edx$timestamp_binary) 
val_by_gen <- cbind(validation[,1:3], genre_matrix_v, validation$timestamp_binary)
colnames(edx_by_gen) <- c("userId", "movieId", "rating", genre_list, "timestamp_binary")
colnames(val_by_gen) <- c("userId", "movieId", "rating", genre_list, "timestamp_binary")
edx_by_gen <- as.matrix(sapply(edx_by_gen, as.numeric))
val_by_gen <- as.matrix(sapply(val_by_gen, as.numeric))


# remove intermediary matrices
rm(genre_matrix, genre_matrix_v, genres, genres_v, genres2, genres2_v)


# Multiply the rating by the OHE for genre
edx_by_gen_mult <- cbind(edx_by_gen[,1:2], edx_by_gen[,"rating"], sweep(edx_by_gen[,4:22], 1, edx_by_gen[,"rating"], "*"), edx_by_gen[,"timestamp_binary"])
val_by_gen_mult <- cbind(val_by_gen[,1:2], val_by_gen[,"rating"], sweep(val_by_gen[,4:22], 1, val_by_gen[,"rating"], "*"), val_by_gen[,"timestamp_binary"])


colnames(edx_by_gen_mult) <- c("userId", "movieId", "rating", "Action", "Adventure", "Animation", "Children",
                               "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                               "Film.Noir", "Horror", "Imax", "Musical", "Mystery","Romance",
                               "Sci.Fi", "Thriller", "War", "Western", "timestamp_binary")

colnames(val_by_gen_mult) <- c("userId", "movieId", "rating", "Action", "Adventure", "Animation", "Children",
                               "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                               "Film.Noir", "Horror", "Imax", "Musical", "Mystery","Romance",
                               "Sci.Fi", "Thriller", "War", "Western", "timestamp_binary")


# Transform the multiplied one-hot-encoded matrix into a user profile for genre.
user_profiles <- edx_by_gen_mult %>%
  as.data.frame() %>%
  group_by(userId) %>%
  summarise(Action_u = mean(Action),
            Adventure_u = mean(Adventure),
            Animation_u = mean(Animation),
            Children_u = mean(Children),
            Comedy_u = mean(Comedy),
            Crime_u = mean(Crime),
            Documentary_u = mean(Documentary),
            Drama_u = mean(Drama),
            Fantasy_u = mean(Fantasy),
            FilmNoir_u = mean(Film.Noir),
            Horror_u = mean(Horror),
            Imax_u = mean(Imax), 
            Musical_u = mean(Musical),
            Mystery_u = mean(Mystery),
            Romance_u = mean(Romance),
            Sci.Fi_u = mean(Sci.Fi),
            Thriller_u = mean(Thriller),
            War_u = mean(War),
            Western_u = mean(Western)) %>%
  as.data.frame()


user_profiles[is.na(user_profiles)] <- 0

# Transform the Test and Validation datasets to include the user profiles
edx_gen_norm <- edx %>%
  left_join(user_profiles, by="userId") %>%
  select(userId, 
         movieId, 
         rating, 
         Action_u, 
         Adventure_u, 
         Animation_u,
         Children_u, 
         Comedy_u,  
         Crime_u,
         Documentary_u, 
         Drama_u,
         Fantasy_u,
         FilmNoir_u,  
         Horror_u, 
         Imax_u,
         Musical_u, 
         Mystery_u, 
         Romance_u, 
         Sci.Fi_u,  
         Thriller_u,  
         War_u, 
         Western_u, 
         timestamp_binary)

val_gen_norm <- validation %>%
  left_join(user_profiles, by="userId") %>%
  select(userId, 
         movieId, 
         rating, 
         Action_u, 
         Adventure_u, 
         Animation_u,
         Children_u, 
         Comedy_u,  
         Crime_u,
         Documentary_u, 
         Drama_u,
         Fantasy_u,
         FilmNoir_u,  
         Horror_u, 
         Imax_u,
         Musical_u, 
         Mystery_u, 
         Romance_u, 
         Sci.Fi_u,  
         Thriller_u,  
         War_u, 
         Western_u, 
         timestamp_binary)


library(h2o)
h2o.init(nthreads = -1, max_mem_size = "8G")



##################
# Define the model in h2o

# turn the matrices into h2o objects
edx_h2o <- as.h2o(edx_gen_norm)
val_h2o <- as.h2o(val_gen_norm)

# Specify labels and predictors
y <- "rating"
x <- setdiff(names(edx_h2o), y)

# Turn the labels into categorical data.
edx_h2o[,y] <- as.factor(edx_h2o[,y])
val_h2o[,y] <- as.factor(val_h2o[,y])

# Train a deep learning model and validate on test set

DL_model <- h2o.deeplearning(
  x = x,
  y = y,
  training_frame = edx_h2o,
  validation_frame = val_h2o,
  distribution = "AUTO",
  activation = "RectifierWithDropout",
  hidden = c(256, 256, 256, 256),
  input_dropout_ratio = 0.15,
  sparse = TRUE,
  epochs = 15,
  stopping_rounds = 5,
  stopping_tolerance = 0.01, #stops if it doesn't improve at least 0.1%
  stopping_metric = "AUTO",
  nfolds = 10,
  variable_importances = TRUE,
  shuffle_training_data = TRUE,
  mini_batch_size = 2000
)



# Get RMSE
DL_RMSE_validation <- h2o.rmse(DL_model, valid = TRUE) # Validation RMSE = 0.8236556
DL_RMSE_training <- h2o.rmse(DL_model) # Train RMSE = 0.8241222

results <- results %>% add_row(model="Deep Neural Network", RMSE=DL_RMSE_validation)


# Making sure the output is presented as required for this code. 
print(results)



h2o.shutdown(prompt = F)

rm(edx_copy, validation_copy)


###########

# The following two commands remove any previously installed H2O packages for R.
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Next, we download packages that H2O depends on.
pkgs <- c("RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}

# Now we download, install and initialize the H2O package for R.
install.packages("h2o", type="source", repos="http://h2o-release.s3.amazonaws.com/h2o/rel-zermelo/4/R")

# Finally, let's load H2O and start up an H2O cluster
library(h2o)
h2o.init(nthreads = -1, max_mem_size = "12G")

##################
# Define the model in h2o

# turn the matrices into h2o objects
edx_h2o <- as.h2o(edx_gen_norm)
val_h2o <- as.h2o(val_gen_norm)

# Specify labels and predictors
y <- "rating"
x <- setdiff(names(edx_h2o), y)

# Turn the labels into categorical data.
edx_h2o[,y] <- as.factor(edx_h2o[,y])
val_h2o[,y] <- as.factor(val_h2o[,y])

# Train a deep learning model and validate on test set

DL_model <- h2o.deeplearning(
  x = x,
  y = y,
  training_frame = edx_h2o,
  validation_frame = val_h2o,
  distribution = "AUTO",
  activation = "RectifierWithDropout",
  hidden = c(256, 256, 256, 256),
  input_dropout_ratio = 0.15,
  sparse = TRUE,
  epochs = 15,
  stopping_rounds = 5,
  stopping_tolerance = 0.01, #stops if it doesn't improve at least 1%
  stopping_metric = "AUTO", # Because it is a classification, the metric is classification_error
  # stopping_metric = "RMSE",
  nfolds = 5,
  variable_importances = TRUE,
  shuffle_training_data = TRUE,
  mini_batch_size = 4000,
  overwrite_with_best_model = TRUE,
  quiet_mode = TRUE,
  l1=1e-5,                        ## add some L1/L2 regularization
  l2=1e-5,
  max_w2=10                       ## helps stability for Rectifier
)



# Save the model
DL_model_path <- h2o.saveModel(object = DL_model, path = getwd(), force =TRUE)

# load the model
DL_model <- h2o.loadModel(DL_model_path)

# Get RMSE
DL_RMSE_validation <- h2o.rmse(DL_model, valid = TRUE) # Validation RMSE = 0.8236556
DL_RMSE_training <- h2o.rmse(DL_model) # Train RMSE = 0.8241222

results <- results %>% add_row(model="Deep Neural Network", RMSE=DL_RMSE_validation)

DL_RMSE_validation
DL_RMSE_training
plot(DL_model)
