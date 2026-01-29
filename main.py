import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
movies = pd.read_csv("IMDb Movies India.csv", encoding='latin1')
movies["Rating"] = movies["Rating"].fillna(movies["Rating"].mean())
movies["Genre"] = movies["Genre"].fillna("Unknown")
movies["Director"] = movies["Director"].fillna("Unknown")
encoder = LabelEncoder()

# Clean Year column: remove parentheses and convert to numeric
movies["Year"] = movies["Year"].str.strip('()').astype(float)
movies["Year"] = movies["Year"].fillna(movies["Year"].median())

# Clean Duration column: remove ' min' and convert to numeric
movies["Duration"] = movies["Duration"].str.replace(' min', '').astype(float)
movies["Duration"] = movies["Duration"].fillna(movies["Duration"].median())

movies["Genre"] = encoder.fit_transform(movies["Genre"])
movies["Director"] = encoder.fit_transform(movies["Director"])
features = movies[["Genre", "Director", "Year", "Duration"]]
target = movies["Rating"]
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=1
)
model = LinearRegression()
model.fit(X_train, y_train)
predicted_ratings = model.predict(X_test)

error = mean_absolute_error(y_test, predicted_ratings)
print("Mean Absolute Error:", error)
sample_movie = pd.DataFrame([[movies["Genre"].iloc[0],
                              movies["Director"].iloc[0],
                              movies["Year"].iloc[0],
                              movies["Duration"].iloc[0]]],
                            columns=["Genre", "Director", "Year", "Duration"])

predicted = model.predict(sample_movie)
print("Predicted rating for sample movie:", predicted[0])
