import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("homework")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    
    with mlflow.start_run():
        max_depth = 10
        random_state = 0 
        
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # Model Training
        rf = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf.predict(X_val)
        
        # Calculate rmse and log it
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        
        # log model
        mlflow.sklearn.log_model(rf, "model")
        
        # Optionally log the model as an artifact
        model_path = os.path.join(data_path, "rf_model.pkl")
        with open(model_path, "wb") as f_out:
            pickle.dump(rf, f_out)
        mlflow.log_artifact(local_path=model_path, artifact_path="artifacts")

        print(f"Model training completed with RMSE: {rmse}")
        

if __name__ == '__main__':
    run_train()