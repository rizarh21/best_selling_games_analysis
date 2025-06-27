import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('bestSelling_games.csv')

print(data.info())

#most downloaded games
most_downloaded_games = data.sort_values(by="estimated_downloads", ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(most_downloaded_games['game_name'], most_downloaded_games['estimated_downloads'], color='skyblue')
plt.xlabel('Downloads')
plt.title('Top 10 Most Downloaded Games')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#highest rating games
highest_rating_games = data.sort_values(by="rating", ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(highest_rating_games['game_name'], highest_rating_games['rating'], color='skyblue')
plt.xlabel('Rating')
plt.title('Top 10 Highest Rating Games')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#top 10 developers with the highest revenue
data['estimated_revenue']= data['estimated_downloads'] * data['price']

top_devp = data.sort_values(by="estimated_revenue", ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_devp['developer'], top_devp['estimated_revenue'], color='skyblue')
plt.xlabel('Estimated Revenue')
plt.title('Top 10 Developers With The Highest Revenue')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


#correlation between price and estimated downloads
correlation = data['price'].corr(data['estimated_downloads'])
print(f"correlation between price and estimated downloads : {correlation:.2f}")

plt.scatter(data['price'], data['estimated_downloads'], alpha=0.4)
plt.xlabel('Price')
plt.ylabel('Estimated Downloads')
plt.title('Price vs Downloads')
plt.show()

#predict estimated downloads
features = ['price', 'rating', 'difficulty', 'length', 'reviews_like_rate', 'all_reviews_number', 'age_restriction']
target = 'estimated_downloads'

x = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.2f}")
print(f"RMSE: {rmse:,.0f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Downloads')
plt.ylabel('Predicted Downloads')
plt.title('Actual vs Predicted Downloads')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.tight_layout()
plt.show()