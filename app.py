from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np
import joblib




app=Flask(__name__)
model=joblib.load('RandomForestRegressionModel.pkl')
car=pd.read_csv('cleaned_car.csv')

@app.route('/')
def index():
    manufacturer=sorted(car['Manufacturer'].unique())
    model=sorted(car['Model'].unique())
    # year=sorted(car['Year'].unique(),reverse=True)
    year=[2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006]
    fuel_type=sorted(car['Fuel_Type'].unique())
    owner_type = car['Owner_Type'].unique()
    return render_template('index.html',Manufacturer=manufacturer, Models=model, Years=year, Fuel_Types=fuel_type,Owner_Type=owner_type)

@app.route('/predict',methods=['POST'])
def predict():

    company=request.form.get('Manufacturer')
    car_model=request.form.get('Model')
    year=2022-int(request.form.get('Year'))
    fuel_type=request.form.get('Fuel-Type')
    driven=int(request.form.get('Kms_Driven'))
    owner=request.form.get('Owner-Type')
    mileage=float(request.form.get('Mileage'))
    engine=float(request.form.get('Engine'))
    power=float(request.form.get('Power'))
    # print(company,car_model,year,driven,fuel_type,owner,mileage,engine,power)
    # return ""

    prediction=model.predict(pd.DataFrame([[company,car_model,year,driven,fuel_type,owner,mileage,engine,power]],columns=['Manufacturer','Model','Year','Kilometers_Driven','Fuel_Type','Owner_Type','Mileage','Engine','Power']
                              ))
    # print(prediction)
    if year>1 & year<=2:
        final=prediction*0.2
    if year>2 & year<=3:
        final=prediction*0.3
    if year>3 & year<=4:
        final=prediction*0.4
    if year>4 & year<=20:
        final=prediction*0.5

    prediction-=final
    
    return str(np.round(prediction[0],2))
if __name__=='__main__':
    app.run(debug=True) 