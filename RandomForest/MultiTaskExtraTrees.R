# Packages
library(extraTrees)
library(dplyr)
library(purrr)
library(readr)
library(tools)
library(tidyr)

# ----------------------------- SETTING UP DATA ----------------------------------------
folder_path <- "C:/Users/human/Documents/W26/Computational Intelligence/Project/awesomeGroupCI/RandomForest/RFInputs/"

file_list <- list.files(
  path = folder_path,
  pattern = "\\.csv$",
  full.names = TRUE
)

data_df <- map_dfr(file_list, function(file) {
  task_name <- substr(file_path_sans_ext(basename(file)), 1, 4)
  
  read_csv(file, show_col_types = FALSE) %>%
    mutate(task = task_name)
}) %>%
  drop_na() %>%
  mutate(
    Date = as.Date(Date),
    task_id = as.integer(as.factor(task))
  ) %>%
  arrange(Date, task)

# ----------------------------- ROLLING CV SETUP ---------------------------------------
# unique dates across all tasks
all_dates <- sort(unique(multi_task_df$Date))

make_rolling_splits_fixed <- function(dates, train_days, val_days, step_days) {
  splits <- list()
  fold <- 1
  
  train_start_idx <- 1
  train_end_idx <- train_days
  
  while ((train_end_idx + val_days) <= length(dates)) {
    train_dates <- dates[train_start_idx:train_end_idx]
    val_dates <- dates[(train_end_idx + 1):(train_end_idx + val_days)]
    
    splits[[fold]] <- list(
      fold = fold,
      train_dates = train_dates,
      val_dates = val_dates,
      train_start = min(train_dates),
      train_end = max(train_dates),
      val_start = min(val_dates),
      val_end = max(val_dates)
    )
    
    train_start_idx <- train_start_idx + step_days
    train_end_idx <- train_end_idx + step_days
    fold <- fold + 1
  }
  
  splits
}

splits <- make_rolling_splits(
  dates = all_dates,
  initial_train_days = 60,# Number of days to train
  val_days = 1, # How many days we want to validate on
  step_days = 20 # How much I jump forward between each window
)

# ----------------------------- RUN CV ---------------------------------------
lag_cols <- paste0("lag_", 1:60)

x_train <- train_data %>%
  select(all_of(lag_cols)) %>%
  data.matrix()

y_train <- as.numeric(train_data$Price)

x_val <- validation_data %>%
  select(all_of(lag_cols)) %>%
  data.matrix()

y_val <- as.numeric(validation_data$Price)

p <- ncol(x_train)

model <- extraTrees(
  x_train,
  y_train,
  nodesize = 3,
  mtry = floor(p / 3),
  numRandomCuts = 2,
  tasks = train_data$task_id
)

preds <- predict(model, x_val, tasks = validation_data$task_id)

print(predict)
