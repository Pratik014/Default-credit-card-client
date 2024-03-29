from flask import Flask,request,render_template,jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application
## Route for a home page

@app.route('/')

def home_page():
    return render_template('home.html') 



@app.route('/predictdefaulter',methods=['GET','POST'])

def predict_defaulter():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            #ID = (request.form.get('ID')),
            LIMIT_BAL = (request.form.get('LIMIT_BAL')),
            AGE = (request.form.get('AGE')),
            BILL_AMT1 = (request.form.get('BILL_AMT1')),
            BILL_AMT2 = (request.form.get('BILL_AMT2')),
            BILL_AMT3 = (request.form.get('BILL_AMT3')),
            BILL_AMT4 = (request.form.get('BILL_AMT4')),
            BILL_AMT5 = (request.form.get('BILL_AMT5')),
            BILL_AMT6 = (request.form.get('BILL_AMT6')),
            PAY_AMT1 = (request.form.get('PAY_AMT1')),
            PAY_AMT2 = (request.form.get('PAY_AMT2')),
            PAY_AMT3 = (request.form.get('PAY_AMT3')),
            PAY_AMT4 = (request.form.get('PAY_AMT4')),
            PAY_AMT5 = (request.form.get('PAY_AMT5')),
            PAY_AMT6 = (request.form.get('PAY_AMT6')),
            SEX = (request.form.get('SEX')),
            EDUCATION = (request.form.get('EDUCATION')),
            MARRIAGE = (request.form.get('MARRIAGE')),


            PAY_1 = (request.form.get('PAY_1')),
            PAY_2 = (request.form.get('PAY_2')),
            PAY_3 = (request.form.get('PAY_3')),
            PAY_4 = (request.form.get('PAY_4')),
            PAY_5 = (request.form.get('PAY_5')),
            PAY_6 = (request.form.get('PAY_6'))

        )


        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predict(pred_df)
        if result != 0:
            result="The person is Defaulter"
        else:
            result="The person is Not Defaulter"


        return render_template('home.html',result=result,pred_df=pred_df)
    
if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)

