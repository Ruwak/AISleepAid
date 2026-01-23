##################################################
# Program: Sleep Buddy (SleepAid)
# Author: Tony, Zac, Mia, Kuwar
# Date: 27/1/2026
# Purpose: To help user analyze and predic sleeping 
#          patterns and habits using ML.
#####################################################

from flask import Flask, flash, request, render_template, redirect, url_for
import json
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder # Tool for encoding string labels into numbers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import random


# Global Variables:
SLEEP_TIME = ["early", "normal", "late"]
AWAKENINGS = ["none", "low", "medium", "high"]
ALCOHOLCONSUMED = ["none", "low", "medium", "high"]
SLEEP_LENGTH = ["short", "medium", "long"]
CAFFEINE_INTAKE = ["none", "low", "medium", "high"]
EFFECTIVENESS = ["true", "false"]

# get file paths
data_file_path = os.path.join(
    os.path.dirname(__file__),
    'static',
    'json',
    'sleep_sessions.json'
)


# Encoders to convert categorical values into numeric labels
mood_encoder = LabelEncoder()
time_encoder = LabelEncoder()
subject_encoder = LabelEncoder()


# run app
app = Flask(__name__)


# Load appointment data
def read_data():
    with open(data_file_path, "r") as f:
        return json.load(f)


# Append new appointment
def append_data(new_data):
    data = read_data()
    data.append(new_data)
    with open(data_file_path, "w") as f:
        json.dump(data, f, indent=4)


# organize and process data
def org_data():
    processed = [] # stores processed data
    
    raw = read_data()
    if not raw:
        return print("No data")

    # add the corresponding numeric values to the features
    for d in raw:
        processed.append({
            "SleepTime": SLEEP_TIME.index(d["SleepTime"]),
            "SleepLength": SLEEP_LENGTH.index(d["SleepLength"]),
            "Awakenings": AWAKENINGS.index(d["Awakenings"]),
            "AlcoholConsumed": ALCOHOLCONSUMED.index(d["AlcoholConsumed"]),
            "CaffeineConsumed": CAFFEINE_INTAKE.index(d["CaffeineConsumed"]),
            "Effectiveness": EFFECTIVENESS.index(d["Effectiveness"])
        })

    # return processed data
    return processed


# find recomendations
def recommend_study_time_brute_force(model, top_n=3):
    top_recommendations = [] # stores top recommendations
    
    for awakenings_i, Awakenings in enumerate(AWAKENINGS):
        for alcoholconsumed_i, alcoholconsumed in enumerate(ALCOHOLCONSUMED):
            for caffeine_i, caffeine in enumerate(CAFFEINE_INTAKE):
                for sleep_time_i, SleepTime in enumerate(SLEEP_TIME):
                    for sleep_length_i, SleepLength in enumerate(SLEEP_LENGTH):

                        # Prepare input for the model
                        x = [[sleep_time_i, sleep_length_i, awakenings_i, alcoholconsumed_i, caffeine_i]]
                        prob = model.predict_proba(x)[0][0]
                        print(prob)

                        candidate = {
                            "SleepTime": SleepTime,
                            "Awakenings": Awakenings,
                            "AlcoholConsumed": alcoholconsumed,
                            "SleepLength": SleepLength,
                            "CaffeineConsumed": caffeine,
                            "probability": prob
                        } # candidate
                        top_recommendations.append(candidate)

                      

    # Sort final results by probability
    top_recommendations.sort(key=lambda r: r["probability"], reverse=True)
    print(top_recommendations)
    top_rec = top_recommendations[:15]
    random.shuffle(top_rec)
   
    return top_rec[:3]




# train decision tree model
def ML_Model():
    data = org_data() # initialize data to the processed sleep data with numeric values
    
    # Separate features and target
    X = [] # features
    y = [] # target
    for d in data:
        X.append([d["SleepTime"], d["SleepLength"], d["Awakenings"], d["AlcoholConsumed"], d["CaffeineConsumed"]])
        y.append(d["Effectiveness"])
    
    # Initialize and train the model
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X, y)
    
    # Print feature importances
    feature_names = ["SleepTime", "SleepLength", "Awakenings", "AlcoholConsumed", "CaffeineConsumed"]
    importances = model.feature_importances_
    for name, imp in zip(feature_names, importances):
        print(f"{name}: {imp*100:.2f}%")

    # Print tree
    tree_text = export_text(model, feature_names=feature_names)
    print(tree_text)

    # return fully trained model
    return model


# compare the accuracy of both models (defunct) ("don't remove because we're going to use something out of there" - Kuwar on this code)
def compare_models():

    # Decision Tree
    dt_data = org_data()
    X_dt = [[d["SleepTime"], d["SleepLength"], d["Awakenings"], d["AlcoholConsumed"], d["CaffeineConsumed"]] for d in dt_data]
    y_dt = [d["Effectiveness"] for d in dt_data]
    X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.2, random_state=42)
    dt_model = DecisionTreeClassifier(max_depth=8)
    dt_model.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

    return dt_model


# Home page
@app.route('/')
def home():
    data = read_data()
    ML_Model()
    return render_template('index.html', data=data)


# Booking page
@app.route('/index.html', methods=["GET", "POST"])
def book():

    if request.method == "POST":

        # set choices to the feature values chosen in the form by the user
        Awakenings = request.form["Awakenings"]
        Effectiveness = request.form["effectiveness"]
        SleepTime = request.form["SleepTime"]
        SleepLength = request.form["SleepLength"]
        CaffeineConsumed = request.form["CaffeineConsumed"]
        AlcoholConsumed = request.form["AlcoholConsumed"]
        
        # store new sleep session
        new_sleep_session = {
            "SleepTime": SleepTime,
            "SleepLength": SleepLength,
            "Awakenings": Awakenings,
            "AlcoholConsumed": AlcoholConsumed,
            "CaffeineConsumed": CaffeineConsumed,
            "Effectiveness": Effectiveness
        } # new_sleep_session
        append_data(new_sleep_session)

        # redirect user back home after they log the data
        return redirect(url_for('home'))

    return render_template(
        "book.html",
        awakeningss=AWAKENINGS,
        effectivenesses=EFFECTIVENESS,
        is_alcoholconsumeded=ALCOHOLCONSUMED,
        caffeine_intakes=CAFFEINE_INTAKE,
        sleep_times=SLEEP_TIME,
        sleep_lengths=SLEEP_LENGTH
    ) # render_template


# recommendations page
@app.route("/recommendations")
def recommendations():

    # Decision Tree
    dt_model = ML_Model()  
    recs_dt = recommend_study_time_brute_force(dt_model, top_n=3)
    dt_features = ["SleepTime", "SleepLength", "Awakenings", "CaffeineConsumed", "AlcoholConsumed"]
    dt_importances = dict(zip(dt_features, dt_model.feature_importances_))
   
    return render_template(
        "reccomendations.html",  
        recs_dt=recs_dt,
        feature_importances_dt=dt_importances,
    ) # render_template


# magic predictor page
@app.route("/magic_predictor", methods=["GET", "POST"])
def magic_predictor():

    # initialize variables
    prediction = None # initialize the prediction to none
    choices = None # initialize the choices to none

    # post predictions
    if request.method == "POST":
        
        # Train model
        model = ML_Model()
        
        # set choices to the feature values chosen in the form by the user
        choices = {
            "SleepTime": request.form["SleepTime"],
            "SleepLength": request.form["SleepLength"],
            "Awakenings": request.form["Awakenings"],
            "CaffeineConsumed": request.form["CaffeineConsumed"],
            "AlcoholConsumed": request.form["AlcoholConsumed"],
        } # choices

        # add the corresponding numeric values to the features
        awakenings_index = AWAKENINGS.index(request.form["Awakenings"])
        SleepTime = SLEEP_TIME.index(request.form["SleepTime"])
        SleepLength = SLEEP_LENGTH.index(request.form["SleepLength"])
        CaffeineConsumed = CAFFEINE_INTAKE.index(request.form["CaffeineConsumed"])
        AlcoholConsumed = ALCOHOLCONSUMED.index(request.form["AlcoholConsumed"])

        x = [[SleepTime, SleepLength, awakenings_index, AlcoholConsumed, CaffeineConsumed]]

        # Predict probability of Effectiveness
        prob = model.predict_proba(x)[0][1]
        prediction = {
            "SleepTime": SleepTime,
            "SleepLength": SleepLength,
            "Awakenings": awakenings_index,
            "CaffeineConsumed": CaffeineConsumed,
            "AlcoholConsumed": AlcoholConsumed,
            "probability": prob
        } # prediction

    return render_template(
        "magic_predictor.html",
        awakeningss=AWAKENINGS,
        is_alcoholconsumeded=ALCOHOLCONSUMED,
        caffeine_intakes=CAFFEINE_INTAKE,
        sleep_times=SLEEP_TIME,
        sleep_lengths=SLEEP_LENGTH,
        prediction=prediction,
        choices=choices
    ) # render_template
    
if __name__ == '__main__':
    app.run(debug=True)
