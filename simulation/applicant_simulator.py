# ============================================================
# APPLICANT SIMULATOR — Credit Risk ML System
# ============================================================

import requests
import random
import time
import os

# ── API URL — local dev vs deployed ──────────────────────────
# Set ENV variable for production: CREDIT_RISK_API_URL=https://your-render-url.onrender.com
API_URL = os.getenv("CREDIT_RISK_API_URL", "http://127.0.0.1:8000") + "/predict"


# ============================================================
# GENERATE SYNTHETIC APPLICANT
# ============================================================

def generate_applicant(scenario: str = "random") -> dict:
    """
    Scenarios:
        random    — mixed realistic applications
        risky     — high-risk applicant profile
        safe      — low-risk applicant profile
    """

    if scenario == "risky":
        return {
            "age":        random.randint(22, 35),
            "income":     random.uniform(15, 40),
            "family":     random.randint(4, 6),
            "ccavg":      random.uniform(5, 15),
            "education":  1,
            "mortgage":   random.uniform(200, 600),
            "online":     0,
            "creditcard": 0,
            "securities_account": 0,
            "cd_account": 0
        }

    elif scenario == "safe":
        return {
            "age":        random.randint(40, 65),
            "income":     random.uniform(80, 200),
            "family":     random.randint(1, 3),
            "ccavg":      random.uniform(0.5, 3.0),
            "education":  random.choice([2, 3]),
            "mortgage":   random.uniform(0, 100),
            "online":     1,
            "creditcard": 1,
            "securities_account": random.choice([0, 1]),
            "cd_account": random.choice([0, 1])
        }

    else:  # random
        return {
            "age":        random.randint(25, 65),
            "income":     random.uniform(20, 150),
            "family":     random.randint(1, 5),
            "ccavg":      random.uniform(0.2, 10.0),
            "education":  random.choice([1, 2, 3]),
            "mortgage":   random.uniform(0, 400),
            "online":     random.choice([0, 1]),
            "creditcard": random.choice([0, 1]),
            "securities_account": random.choice([0, 1]),
            "cd_account": random.choice([0, 1])
        }


# ============================================================
# SEND TO API + PRINT RESULT
# ============================================================

def send_applicant(applicant: dict, idx: int):

    try:
        response = requests.post(API_URL, json=applicant, timeout=10)

        if response.status_code == 200:
            result = response.json()
            print(f"[{idx+1}]  Income={applicant['income']:.1f}K  "
                  f"Mortgage={applicant['mortgage']:.0f}K  "
                  f"Family={applicant['family']}  "
                  f"→  prob={result['risk_probability']:.4f}  "
                  f"band={result['risk_band']}  "
                  f"decision={result['decision']}")
        else:
            print(f"[{idx+1}] API error: {response.status_code}")

    except Exception as e:
        print(f"[{idx+1}] Connection error: {e}")


# ============================================================
# RUN SIMULATION
# ============================================================

def simulate_applications(n: int = 20, scenario: str = "random"):

    print(f"\nSimulating {n} applications  |  scenario={scenario}\n" + "-" * 60)

    for i in range(n):
        applicant = generate_applicant(scenario)
        send_applicant(applicant, i)
        time.sleep(0.5)

    print("-" * 60 + "\nSimulation complete")


if __name__ == "__main__":
    simulate_applications(20, scenario="random")