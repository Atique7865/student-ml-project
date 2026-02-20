import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import mlflow
import dagshub

# connect to DAGsHub
dagshub.init(
    repo_owner="Atique7865",
    repo_name="student-ml-project",
    mlflow=True
)

# dataset
data = {
    "hours": [1,2,3,4,5,6,7,8],
    "result": [35,45,50,60,65,70,80,90]
}

df = pd.DataFrame(data)

X = df[["hours"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

mlflow.start_run()

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)

mlflow.log_param("model", "LinearRegression")
mlflow.log_metric("mse", mse)

mlflow.sklearn.log_model(model, "model")

mlflow.end_run()

print("Training complete")