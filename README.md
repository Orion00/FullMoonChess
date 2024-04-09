# FullMoonChess
BYU 2024 Case Study regarding Chicago crime from 2010-2024

## Research Questions:
- Do Full Moons, Weather, Holidays, or Weekends affect crime in Chicago? How strongly?
- Can we predict accurate daily crime for the coming year?

## Overview of files
`Crimes.csv`,`holidays.csv`, `weather.csv`, `full_moon.csv`, and `US FederalPay and Leave Holidays 2004 to 2100.csv` contain the original data given for the competition

`crimes_cleaned.csv` contains the data after merges and feature engineering in `CleaningFiles.Rmd`

`aggregated_crimes.csv` contains the contents of `crimes_acf` in `analysis.Rmd` and is the dataset used for the ARIMA model (`analysis.Rmd`) and `crime` in `analysis.py` for the Neural Network.

Various plots were created using `analysis.Rmd` for EDA, model validation, and predictions.

## Findings
Unfortunately (or fortunately), we found no evidence of criminal werewolves in Chicago.