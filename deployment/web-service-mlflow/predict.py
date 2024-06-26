import mlflow
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
RUN_ID = "5b2a074db3d04af393592de8aa00301d"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


logged_model = f"mlflow-artifacts:/2/{RUN_ID}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    features = {}
    features["PU_DO"] = "%s_%s" % (ride["PULocationID"], ride["DOLocationID"])
    features["trip_distance"] = ride["trip_distance"]
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask("duration-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    preds = predict(features)

    result = {"duration": preds, "model_version": RUN_ID}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
