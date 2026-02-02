def heart_risk_explanation(inputs):
    explanation = []

    if inputs["ejection_fraction"] < 40:
        explanation.append("Low ejection fraction")
    if inputs["serum_creatinine"] > 1.5:
        explanation.append("High serum creatinine")
    if inputs["high_blood_pressure"] == 1:
        explanation.append("High blood pressure")
    if inputs["smoking"] == 1:
        explanation.append("Smoking habit detected")

    return explanation
