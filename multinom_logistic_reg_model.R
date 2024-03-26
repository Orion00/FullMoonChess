library(forecast)
library(tseries)
library(dplyr)
library(nnet)

# Assuming 'crimes_cleaned.csv' has a 'Date' and 'Primary.Type' column,
# and you've converted 'Date' to a Date or POSIXct format during read.
crimes <- read.csv("C:/Users/jbren/Documents/STAT/FullMoonChess/crimes_cleaned.csv")

crimes$new_date

crimes$Primary.Type <- as.factor(crimes$Primary.Type)
crimes$new_date <- as.POSIXct(crimes$new_date, format = "%Y-%m-%d")
crimes$Time <- as.POSIXct(crimes$Time, format = "%H:%M:%S")

crimes <- crimes %>%
  mutate(
    Month = as.factor(format(new_date, "%m")), # Ensuring factor
    Hour = as.factor(format(new_date, "%H")), # Ensuring factor
    DayOfWeek = as.factor(weekdays(new_date)),
    IsWeekend = as.factor(ifelse(DayOfWeek %in% c('Saturday', 'Sunday'), 'Yes', 'No'))
  )

# Check for factors with 2 or more levels and prepare for dynamic formula
valid_factors <- sapply(crimes[, c("Month", "Hour", "DayOfWeek", "IsWeekend")], function(x) nlevels(x) >= 2)

# Keep only valid factors
valid_factor_names <- names(valid_factors)[valid_factors]

# Ensure there are valid factors to avoid formula errors
if (length(valid_factor_names) > 0) {
  # Constructing the formula dynamically
  formula_str <- paste("Primary.Type ~", paste(valid_factor_names, collapse = " + "))
  formula <- as.formula(formula_str)
  
  # Fit the multinomial logistic regression model
  model <- multinom(formula, data = crimes)
  
  # Summary of the model
  summary(model)
} else {
  print("No valid factors with 2 or more levels were found for model fitting.")
}

