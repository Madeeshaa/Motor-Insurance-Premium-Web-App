import re

with open('premium_app_script.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Improve the GBM Stacker hyperparameters to boost Gini
text = re.sub(
    r'gb_stack = GradientBoostingRegressor\(\s*n_estimators=500,\s*max_depth=4,\s*learning_rate=0.05,\s*subsample=0.8,\s*min_samples_leaf=50,\s*loss="huber",\s*random_state=42,\s*\)',
    'gb_stack = GradientBoostingRegressor(\n    n_estimators=1500,\n    max_depth=5,\n    learning_rate=0.015,\n    subsample=0.75,\n    min_samples_leaf=30,\n    loss="huber",\n    random_state=42,\n)',
    text
)

# 2. Extract the streamlit code at the end
split_str = 'pip install streamlit streamlit-jupyter'
parts = text.split(split_str)

if len(parts) > 1:
    training_code = parts[0]
else:
    training_code = text

# We will write the Streamlit app to app.py
# that imports premium_app_script (which will train models and make them accessible).
# So we need to ensure the end of premium_app_script has a prediction function.

footer = """
# Prediction function for the Streamlit UI
def predict_pure_premium(age, exp, v_age, v_value, cc):
    # Prepare standard features
    X_single = np.array([[age, exp, v_age, v_value, cc]], dtype=np.float32)
    
    # 1. NF prediction
    nf_lam_val, nf_sev_val = nf_predict(X_single)
    nf_pp = float(nf_lam_val[0] * nf_sev_val[0])
    
    # 2. GLM prediction
    # Needs a DataFrame for statsmodels
    df_single = pd.DataFrame([{
        "Policyholder_age": age,
        "Driving_Experience": exp,
        "Vehicle_age": v_age,
        "Value_vehicle": v_value,
        "Cylinder_capacity": cc
    }])
    glm_freq_val = float(nb_model.predict(df_single).iloc[0])
    glm_sev_val = float(gamma_model.predict(df_single).iloc[0])
    glm_pp = glm_freq_val * glm_sev_val
    
    # 3. Stack Features
    stack_X = build_stack_features(pd.DataFrame({
        "glm_pure_premium": [glm_pp],
        "nf_pure_premium": [nf_pp],
        "glm_freq": [glm_freq_val],
        "nf_freq": [nf_lam_val[0]],
    }))
    stack_X = np.nan_to_num(stack_X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 4. GBM Stacker predictor
    stack_raw_pred = gb_stack.predict(stack_X)
    stack_raw_pred_clip = np.clip(stack_raw_pred, 0, None)
    
    # 5. Isotonic calibration
    final_pp = iso.predict(stack_raw_pred_clip)[0]
    return float(np.clip(final_pp, 0, None))
"""

with open('premium_app_script.py', 'w', encoding='utf-8') as f:
    f.write(training_code + footer)

