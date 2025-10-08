# Best-Selling Steam Games of All Time Analysis 

This project analyzes and visualizes the best-selling games data on the Steam platform. Special thanks to H. Buğra Eken, Brian Risk, and KaichiBL for this amazing dataset that contains 2,380 games data which were collected on June 1, 2025, from the official 'Bestsellers' page on the Steam store. You can find this dataset at kaggle.com/datasets/hbugrae/best-selling-steam-games-of-all-time

## Goal
Analyzed the best-selling game dataset on the Steam platform to understand the factors that influence the number of downloads and predict download estimates using an XGBoost Regressor-based machine learning model.

## Key Objectives
- Determine the top 10 most downloaded games and highest-rated games
- Analyzed top 10 highest developers estimated revenue
- Find out the correlation between price and estimated downloads
- Create modelling about predicted vs actual downloads
- Visualization data by using Matplotlib

## Tools
- Python
- Pandas
- Matplotlib
- Scikit Learn
- XGBoost

## Insights
- There is no strong correlation between game price and number of downloads, indicating that other factors (such as ratings and popularity) have a greater influence.
- Some premium games still have high download rates despite their high prices.
- The XGBoost model provides good prediction performance with a high R² value, demonstrating its ability to capture non-linear patterns between features and targets.
