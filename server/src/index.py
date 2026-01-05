from flask import Flask, jsonify, request
from flask_cors import CORS
from models import predictions
from bands.read_bands_webapp import combine_bands, plot


app = Flask(__name__)
CORS(app)

pred = predictions()
loaded_models = pred.load_models()

@app.route("/predict",methods=['GET'])
def get_predict():

    inputs_str = request.args.get("inputs", "")
    inputs = [float(x) for x in inputs_str.split(",")]

    graph_type = request.args.get("graph_type", "")

    result_xs,result_ys = pred.predict(inputs,loaded_models,graph_type)

    if hasattr(result_xs, "tolist"):
        result_xs = result_xs.tolist()

    if hasattr(result_ys, "tolist"):
        result_ys = result_ys[0].tolist()

    print({"xs": result_xs, "ys": result_ys})

    return jsonify({"xs": result_xs, "ys": result_ys})

@app.route("/getBands",methods=['GET'])
def get_bands():
    telescopes_str = request.args.get("telescopes", "")
    telescopes = telescopes_str.split(",")

    redshift = float(request.args.get("redshift", "0"))
    bands = combine_bands(telescopes)

    datasets = plot(bands, redshift=redshift)

    for ds in datasets:
        if hasattr(ds["xs"], "tolist"):
            ds["xs"] = ds["xs"].tolist()
        if hasattr(ds["ys"], "tolist"):
            ds["ys"] = ds["ys"].tolist()

    print("OUTPUT")
    print({"datasets": datasets})

    return jsonify({"datasets": datasets})

if __name__ == "__main__":
    app.run()