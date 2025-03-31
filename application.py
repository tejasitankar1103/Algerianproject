from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

##import ridge regressor and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_Scalar=pickle.load(open('models/scaler.pkl','rb'))




@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature = float(request.form.get('temperature'))  # get k baad '' esme vo chahiye jo
        RH = float(request.form.get('rh'))           #html k form mai name mai likha hai har parameter k liye
        Ws=float(request.form.get('ws'))
        Rain=float(request.form.get('rain'))
        FFMC=float(request.form.get('ffmc'))
        DMC=float(request.form.get('dmc'))
        ISI=float(request.form.get('isi'))
        Classes=float(request.form.get('classes'))
        Region=float(request.form.get('region'))

        new_data_scaled = standard_Scalar.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])



    else:
        return render_template('home.html')

if __name__ =="__main__":
    app.run(host="0.0.0.0")