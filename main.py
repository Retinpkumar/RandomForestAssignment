from flask import Flask, render_template, request
from flask_cors import cross_origin
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__, template_folder='templates')

@app.route('/')
@cross_origin()
def home_page():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
@cross_origin()
def result_page():
    if request.method == 'POST':
        try:
            crim = float(request.form['crim'])
            zn = float(request.form['zn'])
            age = float(request.form['age'])
            dis = float(request.form['dis'])
            rad = float(request.form['rad'])
            tax = float(request.form['tax'])
            b = float(request.form['b'])
            lstat = float(request.form['lstat'])
            indus = float(request.form['indus'])
            rm = float(request.form['rm'])
        except Exception as e:
            print(e)
            raise Exception()

        try:
            df_pred = pd.DataFrame({"CRIM": crim,
                                    "ZN": zn,
                                    "AGE": age,
                                    "DIS": dis,
                                    "RAD": rad,
                                    "TAX": tax,
                                    "B": b,
                                    "LSTAT": lstat}, index=[1])

            # Input for RAD
            if rad == 24:
                df_pred['RAD'] = 1
            else:
                df_pred['RAD'] = 0

            # Input for B
            if b < 380:
                df_pred['B'] = 0
            else:
                df_pred['B'] = 1

            # Input for INDUS

            df_pred['INDUS_1'] = 0
            df_pred['INDUS_2'] = 0
            df_pred['INDUS_3'] = 0
            df_pred['INDUS_4'] = 0
            df_pred['INDUS_5'] = 0
            df_pred['INDUS_6'] = 0
            df_pred['INDUS_7'] = 0
            df_pred['INDUS_8'] = 0
            df_pred['INDUS_9'] = 0
            df_pred['INDUS_10'] = 0
            df_pred['INDUS_11'] = 0
            df_pred['INDUS_12'] = 0
            df_pred['INDUS_13'] = 0
            df_pred['INDUS_15'] = 0
            df_pred['INDUS_18'] = 0
            df_pred['INDUS_19'] = 0
            df_pred['INDUS_21'] = 0
            df_pred['INDUS_25'] = 0
            df_pred['INDUS_27'] = 0

            if indus == 1:
                df_pred['INDUS_1'] = 1
            elif indus == 2:
                df_pred['INDUS_2'] = 1
            elif indus == 3:
                df_pred['INDUS_3'] = 1
            elif indus == 4:
                df_pred['INDUS_4'] = 1
            elif indus == 5:
                df_pred['INDUS_5'] = 1
            elif indus == 6:
                df_pred['INDUS_6'] = 1
            elif indus == 7:
                df_pred['INDUS_7'] = 1
            elif indus == 8:
                df_pred['INDUS_8'] = 1
            elif indus == 9:
                df_pred['INDUS_9'] = 1
            elif indus == 10:
                df_pred['INDUS_10'] = 1
            elif indus == 11:
                df_pred['INDUS_11'] = 1
            elif indus == 12:
                df_pred['INDUS_12'] = 1
            elif indus == 13:
                df_pred['INDUS_13'] = 1
            elif indus == 15:
                df_pred['INDUS_15'] = 1
            elif indus == 18:
                df_pred['INDUS_18'] = 1
            elif indus == 19:
                df_pred['INDUS_19'] = 1
            elif indus == 21:
                df_pred['INDUS_21'] = 1
            elif indus == 25:
                df_pred['INDUS_25'] = 1
            elif indus == 27:
                df_pred['INDUS_27'] = 1
            else:
                df_pred['INDUS_1'] = 0
                df_pred['INDUS_2'] = 0
                df_pred['INDUS_3'] = 0
                df_pred['INDUS_4'] = 0
                df_pred['INDUS_5'] = 0
                df_pred['INDUS_6'] = 0
                df_pred['INDUS_7'] = 0
                df_pred['INDUS_8'] = 0
                df_pred['INDUS_9'] = 0
                df_pred['INDUS_10'] = 0
                df_pred['INDUS_11'] = 0
                df_pred['INDUS_12'] = 0
                df_pred['INDUS_13'] = 0
                df_pred['INDUS_15'] = 0
                df_pred['INDUS_18'] = 0
                df_pred['INDUS_19'] = 0
                df_pred['INDUS_21'] = 0
                df_pred['INDUS_25'] = 0
                df_pred['INDUS_27'] = 0

            # Input for RM

            df_pred['RM_4'] = 0
            df_pred['RM_5'] = 0
            df_pred['RM_6'] = 0
            df_pred['RM_7'] = 0
            df_pred['RM_8'] = 0

            if rm == 4:
                df_pred['RM_4'] = 1
            elif rm == 5:
                df_pred['RM_5'] = 1
            elif rm == 6:
                df_pred['RM_6'] = 1
            elif rm == 7:
                df_pred['RM_7'] = 1
            elif rm == 8:
                df_pred['RM_8'] = 1
            else:
                df_pred['RM_4'] = 0
                df_pred['RM_5'] = 0
                df_pred['RM_6'] = 0
                df_pred['RM_7'] = 0
                df_pred['RM_8'] = 0
        except Exception as e:
            print(e)
            raise Exception()

        boxcox_feat=['CRIM','DIS','LSTAT']

        for feat in boxcox_feat:
            df_pred[feat] = np.log1p(df_pred[feat])

        try:
            # Standardizing the data
            scaler_file = 'standard_scaler.pickle'
            scaled_model = pickle.load(open(scaler_file, 'rb'))
        except Exception as e:
            print(e)
            raise Exception()

        try:
            df_test_scaled = scaled_model.transform(df_pred)
        except Exception as e:
            print(e)
            raise Exception()

        try:
            model_file = 'random_forest_model.pickle'
            loaded_model = pickle.load(open(model_file, 'rb'))
        except Exception as e:
            print(e)
            raise Exception()

        try:
            prediction = loaded_model.predict(df_test_scaled)
            print("Prediction is :", prediction)
            return render_template("result.html", prediction=prediction[0].round(2))
        except Exception as e:
            print(e)
            raise Exception()
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
