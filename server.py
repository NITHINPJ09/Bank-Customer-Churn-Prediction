from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    geography = request.form['geography']
    creditscore = int(request.form['creditScore'])
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    tenure = int(request.form['tenure'])
    balance = float(request.form['balance'])
    numofproducts = int(request.form['numofproducts'])
    hascrcard = int(request.form['hascrcard'])
    isactivemember = int(request.form['isactivemember'])
    estimatedsalary = float(request.form['estimatedsalary'])

    g1 = int(geography[0])
    g2 = int(geography[2])
    g3 = int(geography[4])

    scaler = joblib.load('scaler.pkl')
    model = joblib.load('churn_predict_model.pkl')

    pred = model.predict(scaler.transform([[creditscore, age, tenure, balance, numofproducts, hascrcard, isactivemember, estimatedsalary, g2, g3, gender]]))
    prediction = int(pred[0])
    print(prediction)

    if prediction == 1:
        result = "The customer is unlikely to remain with the bank."
    else:
        result = "The customer is likely to stay with the bank."


    return render_template('predict.html', prediction_text=result)