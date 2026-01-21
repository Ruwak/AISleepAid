import json
import random
import numpy as np

SUBJECTS = ["Law", "Physics", "Computer Science", "Pre-calc"]
TIMES_OF_DAY = ["Morning", "Afternoon", "Evening", "Night"]

TOTAL_ENTRIES = 300000

ENERGY_MAP = {
    1: "Low",
    2: "Medium",
    3: "High"
}

# -------------------------------
# PERSON PROFILE (habits)
# -------------------------------
SUBJECT_WEIGHTS = {
    "Physics": 0.35,
    "Computer Science": 0.35,
    "Law": 0.2,
    "Pre-calc": 0.1
}

TIME_WEIGHTS = {
    "Morning": 0.1,
    "Afternoon": 0.2,
    "Evening": 0.5,
    "Night": 0.2
}

AVERAGE_STUDY_TIME = 110
STUDY_TIME_STD = 25

SUBJECT_PROFILE = {
    "Physics": {"energy_boost": 1, "effectiveness_bias": 0.15},
    "Computer Science": {"energy_boost": 1, "effectiveness_bias": 0.15},
    "Law": {"energy_boost": 0, "effectiveness_bias": 0.05},
    "Pre-calc": {"energy_boost": -1, "effectiveness_bias": -0.1}
}

def weighted_choice(weights):
    return random.choices(
        list(weights.keys()),
        weights=list(weights.values()),
        k=1
    )[0]

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def generate_session():
    subject = weighted_choice(SUBJECT_WEIGHTS)
    time_of_day = weighted_choice(TIME_WEIGHTS)

    # Clustered study time
    time_spent = int(np.random.normal(AVERAGE_STUDY_TIME, STUDY_TIME_STD))
    time_spent = clamp(time_spent, 20, 240)

    # Energy
    base_energy = random.choice([2, 3])
    energy = clamp(
        base_energy + SUBJECT_PROFILE[subject]["energy_boost"],
        1,
        3
    )

    # Effectiveness probability
    effectiveness_prob = (
        0.65
        + SUBJECT_PROFILE[subject]["effectiveness_bias"]
        + (0.1 if time_of_day == "Evening" else -0.05)
        + (0.05 if energy == 3 else -0.05)
    )

    effective = random.random() < effectiveness_prob

    # Random off-days (10%)
    if random.random() < 0.1:
        energy = random.choice([1, 2])
        time_spent = random.randint(15, 60)
        effective = False

    return {
        "subject": str(subject),
        "time": str(time_spent),
        "time_of_day": str(time_of_day),
        "energy_level": ENERGY_MAP[energy],
        "effectiveness": "Yes" if effective else "No"
    }

# -------------------------------
# Generate dataset
# -------------------------------
data = [generate_session() for _ in range(TOTAL_ENTRIES)]

with open("study_sessions.json", "w") as f:
    json.dump(data, f, indent=2)

print("Generated realistic habit-based study session data (all strings)")
