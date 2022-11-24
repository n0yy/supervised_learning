# Import Library/Module
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# Read data and drop table
df = pd.read_csv("data/titanic.csv", index_col="PassengerId")
df.drop(columns=["Name", "Age", "Ticket", "Cabin"], inplace=True)

# Splitting Data
X = df.drop(columns="Survived")
y = df["Survived"]

"""
    1/3 = Data Testing
    2/3 = Data Training
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    stratify=y, 
                                                    random_state=42
                                    )
                                    
# Preprocessor
numeric_data = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", MinMaxScaler())
])

categoric_data = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot", OneHotEncoder())
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_data, ["Fare", "SibSp", "Parch"]),
    ("categoric", categoric_data, ["Pclass", "Sex", "Embarked"])
])

# Main Pipeline
pipeline = Pipeline([
    ("prep", preprocessor),
    ("algo", KNeighborsClassifier())
])

# Parameter Tuning
model = GridSearchCV(
                pipeline,
                param_grid={
                    "algo__n_neighbors": range(1, 51, 2),
                    "algo__weights": ["uniform", "distance"],
                    "algo__p": [1, 2],
                },
                cv=5,
                n_jobs=-1,
                verbose=1
)
model.fit(X_train, y_train)


# Evaluation
print(model.best_params_)
print(model.score(X_train, y_train), model.score(X_test, y_test))

# Prediction
data = [
    [1, "female", 1, 1, 100, "S"],
    [3, "male", 0, 0, 10, "S"]
]
X_pred = pd.DataFrame(data, index=["Rose", "Jack"], columns=X.columns)
print(model.predict(X_pred))