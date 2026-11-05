import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'model.sav')
model = pickle.load(open(model_path, 'rb'))

csv_path = os.path.join(BASE_DIR, 'first_telc.csv')
df1 = pd.read_csv(csv_path)


@app.route("/")
def loadPage():
    return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    try:
        inputQuery1 = request.form['query1']
        inputQuery2 = request.form['query2']
        inputQuery3 = request.form['query3']
        inputQuery4 = request.form['query4']
        inputQuery5 = request.form['query5']
        inputQuery6 = request.form['query6']
        inputQuery7 = request.form['query7']
        inputQuery8 = request.form['query8']
        inputQuery9 = request.form['query9']
        inputQuery10 = request.form['query10']
        inputQuery11 = request.form['query11']
        inputQuery12 = request.form['query12']
        inputQuery13 = request.form['query13']
        inputQuery14 = request.form['query14']
        inputQuery15 = request.form['query15']
        inputQuery16 = request.form['query16']
        inputQuery17 = request.form['query17']
        inputQuery18 = request.form['query18']
        inputQuery19 = request.form['query19']

        data = [[
            int(inputQuery1),
            float(inputQuery2),
            float(inputQuery3),
            inputQuery4,
            inputQuery5,
            inputQuery6,
            inputQuery7,
            inputQuery8,
            inputQuery9,
            inputQuery10,
            inputQuery11,
            inputQuery12,
            inputQuery13,
            inputQuery14,
            inputQuery15,
            inputQuery16,
            inputQuery17,
            inputQuery18,
            int(inputQuery19)
        ]]

        new_df = pd.DataFrame(data, columns=[
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
            'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'tenure'
        ])

        df_2 = pd.concat([df1, new_df], ignore_index=True)

        labels = ["{0}-{1}".format(i, i + 11) for i in range(1, 72, 12)]
        df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
        df_2.drop(columns=['tenure'], inplace=True)

        df_dummies = pd.get_dummies(df_2)

        if hasattr(model, 'feature_names_in_'):
            df_dummies = df_dummies.reindex(columns=model.feature_names_in_, fill_value=0)

        single = model.predict(df_dummies.tail(1))
        probability = model.predict_proba(df_dummies.tail(1))[:, 1]
        confidence = round(float(probability[0]) * 100, 2)

        if single[0] == 1:
            o1 = "This customer is likely to churn!!"
            o2 = f"Confidence: {confidence}%"
        else:
            o1 = "This customer is likely to continue!!"
            o2 = f"Confidence: {confidence}%"

        return render_template('home.html', output1=o1, output2=o2,
                               query1=inputQuery1, query2=inputQuery2, query3=inputQuery3,
                               query4=inputQuery4, query5=inputQuery5, query6=inputQuery6,
                               query7=inputQuery7, query8=inputQuery8, query9=inputQuery9,
                               query10=inputQuery10, query11=inputQuery11, query12=inputQuery12,
                               query13=inputQuery13, query14=inputQuery14, query15=inputQuery15,
                               query16=inputQuery16, query17=inputQuery17, query18=inputQuery18,
                               query19=inputQuery19)

    except Exception as e:
        return render_template('home.html', output1="Error Processing Input!", output2=str(e))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
