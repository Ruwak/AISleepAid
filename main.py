from flask import Flask, flash, request, render_template, redirect, url_for
import json
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Tool for encoding string labels into numbers
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# import pandas as pd
import os



# Global Variables:
SLEEP_TIME = ["early", "normal", "late"]
AWAKENINGS = ["none", "low", "medium", "high"]
ALCOHOLCONSUMED = ["none", "low", "medium", "high"]
SLEEP_LENGTH = ["Short", "Medium", "Long"]
CAFFEINE_INTAKE = ["none", "low", "medium", "high"]
EFFECTIVENESS = ["true", "false"]
data_file_path = os.path.join(
    os.path.dirname(__file__),
    'static',
    'json',
    'sleep_sessions.json'
)
# csv_path = os.path.join(
#     os.path.dirname(__file__),
#     'static',
#     'data',
#     'sleep_sessions.csv'
# )

# pd.read_csv(csv_path)[
#     ["Effectiveness", "Awakenings", "CaffeineConsumed", "SleepLength", "AlcoholConsumed","SleepTime"]
# ].to_json(data_file_path, orient="records", indent=4)


# Encoders to convert categorical values into numeric labels
mood_encoder = LabelEncoder()
time_encoder = LabelEncoder()
subject_encoder = LabelEncoder()


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

 
    


def org_data():
    

    raw = read_data()
    if not raw:
        return print("No data")
    
    processed = []

    for d in raw:
        # "Effectiveness", "Awakenings", "CaffeineConsumed", "SleepLength", "AlcoholConsumed","SleepTime"
        processed.append({
            "SleepTime": SLEEP_TIME.index(d["SleepTime"]),
            "SleepLength": SLEEP_LENGTH.index(d["SleepLength"]),
            "Awakenings": AWAKENINGS.index(d["Awakenings"]),
            "AlcoholConsumed": ALCOHOLCONSUMED.index(d["AlcoholConsumed"]),
            "CaffeineConsumed": CAFFEINE_INTAKE.index(d["CaffeineConsumed"]),
            "Effectiveness": EFFECTIVENESS.index(d["Effectiveness"])
        })

    return processed



def recommend_study_time_brute_force(model, top_n=3):

    top_recommendations = []

    
    for awakenings_i, Awakenings in enumerate(AWAKENINGS):
        for alcoholconsumed_i, alcoholconsumed in enumerate(ALCOHOLCONSUMED):
            for caffeine_i, caffeine in enumerate(CAFFEINE_INTAKE):
                for sleep_time_i, SleepTime in enumerate(SLEEP_TIME):
                    for sleep_length_i, SleepLength in enumerate(SLEEP_LENGTH):

            

                        # Prepare input for the model
                        x = [[sleep_time_i, awakenings_i, alcoholconsumed_i, sleep_length_i, caffeine_i]]
                        prob = model.predict_proba(x)[0][1]

                        candidate = {
                            "SleepTime": SleepTime,
                            "Awakenings": Awakenings,
                            "AlcoholConsumed": alcoholconsumed,
                            "SleepLength": SleepLength,
                            "CaffeineConsumed": caffeine,
                            "probability": prob
                        }

                        # Fill initial slots
                        if len(top_recommendations) < top_n:
                            top_recommendations.append(candidate)
                            continue

                        # Find lowest probability in current top list
                        min_prob = min(r["probability"] for r in top_recommendations)

                        # Replace if better
                        if prob > min_prob:
                            idx = next(
                                i for i, r in enumerate(top_recommendations)
                                if r["probability"] == min_prob
                            )
                            top_recommendations[idx] = candidate

    # Sort final results by probability
    top_recommendations.sort(key=lambda r: r["probability"], reverse=True)
    print(top_recommendations)

    return top_recommendations


def ML_Model():
    data = org_data()
    
    # Separate features and target
    X = []
    y = []
    for d in data:
        X.append([d["SleepTime"], d["SleepLength"], d["Awakenings"], d["AlcoholConsumed"], d["CaffeineConsumed"]])
        y.append(d["Effectiveness"])
    
    # Initialize and train the model
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    # Print feature importances
    feature_names = ["SleepTime", "SleepLength", "Awakenings", "AlcoholConsumed", "CaffeineConsumed"]
    importances = model.feature_importances_
    for name, imp in zip(feature_names, importances):
        print(f"{name}: {imp*100:.2f}%")

    # Print tree
    tree_text = export_text(model, feature_names=feature_names)
    print(tree_text)

    return model

# Example usage:
# model = ML_Model()

# print(generate_recommendation(model))

def compare_models():

    # Decision Tree
    dt_data = org_data()
    X_dt = [[d["SleepTime"], d["SleepLength"], d["Awakenings"], d["AlcoholConsumed"], d["CaffeineConsumed"]] for d in dt_data]
    y_dt = [d["Effectiveness"] for d in dt_data]
    X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.2, random_state=42)
    dt_model = DecisionTreeClassifier(max_depth=8)
    dt_model.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))


    print(f"Decision Tree Accuracy: {dt_acc*100:.2f}%")

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
        Awakenings = request.form["Awakenings"]
        Effectiveness = request.form["effectiveness"]
        SleepTime = request.form["SleepTime"]
        SleepLength = request.form["SleepLength"]
        CaffeineConsumed = request.form["CaffeineConsumed"]
        AlcoholConsumed = request.form["AlcoholConsumed"]
        
        new_sleep_session = {
            "SleepTime": SleepTime,
            "SleepLength": SleepLength,
            "Awakenings": Awakenings,
            "AlcoholConsumed": AlcoholConsumed,
            "CaffeineConsumed": CaffeineConsumed,
            "Effectiveness": Effectiveness
        }
        append_data(new_sleep_session)
        return redirect(url_for('home'))

    return render_template(
        "book.html",
        awakeningss=AWAKENINGS,
        effectivenesses=EFFECTIVENESS,
        is_alcoholconsumeded=ALCOHOLCONSUMED,
        caffeine_intakes=CAFFEINE_INTAKE,
        sleep_times=SLEEP_TIME,
        sleep_lengths=SLEEP_LENGTH
    )

@app.route("/recommendations")
def recommendations():
    # Decision Tree
    dt_model = ML_Model()  
    recs_dt = recommend_study_time_brute_force(dt_model, top_n=3)
    dt_features = ["SleepTime", "SleepLength", "Awakenings", "CaffeineConsumed", "AlcoholConsumed"]
    dt_importances = dict(zip(dt_features, dt_model.feature_importances_))
    print("helloWorld")
   
    return render_template(
        "reccomendations.html",  
        recs_dt=recs_dt,
        feature_importances_dt=dt_importances,
    )

@app.route("/magic_predictor", methods=["GET", "POST"])
def magic_predictor():

    prediction = None

    if request.method == "POST":
        # Train model
        model = ML_Model()

        # Encode input
        awakenings_index = AWAKENINGS.index(request.form["Awakenings"])
        SleepTime = int(request.form["SleepTime"])
        SleepLength = int(request.form["SleepLength"])
        CaffeineConsumed = int(request.form["CaffeineConsumed"])
        AlcoholConsumed = int(request.form["AlcoholConsumed"])

        x = [[awakenings_index, SleepTime, SleepLength, CaffeineConsumed, AlcoholConsumed]]

        # Predict probability of Effectiveness
        prob = model.predict_proba(x)[0][1]

        prediction = {
            "SleepTime": SleepTime,
            "SleepLength": SleepLength,
            "Awakenings": request.form["Awakenings"],
            "CaffeineConsumed": CaffeineConsumed,
            "AlcoholConsumed": AlcoholConsumed,
            "probability": prob
        }

    return render_template(
        "magic_predictor.html",
        awakeningss=AWAKENINGS,
        prediction=prediction
    )
    
@app.route('/test-ai')
def test_ai():
    data = org_data()

    X = [[d["SleepTime"], d["SleepLength"], d["Awakenings"], d["CaffeineConsumed"], d["AlcoholConsumed"]] for d in data]
    y = [d["Effectiveness"] for d in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    accuracy = round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)

    X_dt = [[d["SleepTime"], d["SleepLength"], d["Awakenings"], d["CaffeineConsumed"], d["AlcoholConsumed"]] for d in data]
    y_dt = [d["Effectiveness"] for d in data]
    X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.2, random_state=42)
    dt_model = DecisionTreeClassifier(max_depth=8)
    dt_model.fit(X_train, y_train)
    accuracy2 = round(accuracy_score(y_test, dt_model.predict(X_test)) * 100, 2) 

    return render_template(
        'test_ai_accuracy.html',
        accuracy=accuracy,
        accuracy2=accuracy2,
        total=len(data)
    )

# What Mo said
if __name__ == '__main__':
    app.run(debug=True)