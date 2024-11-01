from flask import Flask, request, jsonify
from load_pickle import dv, model

app = Flask('predict_subscription')

@app.route('/')
def home():
    return 'Hello!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    data_encoded = dv.transform(data)
    y_pred = model.predict_proba(data_encoded)[0, 1]

    return jsonify({
        'subscription_probability': y_pred
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)