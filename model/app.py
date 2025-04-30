import streamlit as st
import numpy as np
import pickle

# --- Load trained model and encoders ---
model = pickle.load(open("model/trained_model_rf.sav", "rb"))
le_category = pickle.load(open("model/category_encoder.pkl", "rb"))
le_gender = pickle.load(open("model/gender_encoder.pkl", "rb"))
le_city = pickle.load(open("model/city_encoder.pkl", "rb"))
le_state = pickle.load(open("model/state_encoder.pkl", "rb"))


# --- Helper: Create dropdown options like "shopping_pos (0)" ---
def make_options_dict(encoder):
    return {f"{cls} ({encoder.transform([cls])[0]})": cls for cls in encoder.classes_}

# --- Prediction function ---
def fraud_prediction(input_data):
    input_np = np.asarray(input_data, dtype=float).reshape(1, -1)
    pred = model.predict(input_np)
    return '✅ Legit transaction' if pred[0] == 0 else '🚨 Fraudulent transaction'

# --- Main App ---
def main():
    st.title("💳 Credit Card Fraud Detection")

    # Dropdown options with labels and encoded values
    category_options = make_options_dict(le_category)
    gender_options = make_options_dict(le_gender)
    city_options = make_options_dict(le_city)
    state_options = make_options_dict(le_state)

    # --- User inputs ---
    selected_category = st.selectbox("Transaction Category", list(category_options.keys()))
    selected_gender = st.selectbox("Card Holder Gender", list(gender_options.keys()))
    selected_city = st.selectbox("City", list(city_options.keys()))
    selected_state = st.selectbox("State", list(state_options.keys()))

    amt = st.text_input("Transaction Amount", "100.0")
    lat = st.text_input("Latitude", "40.7128")
    long = st.text_input("Longitude", "-74.0060")
    city_pop = st.text_input("City Population", "8000000")

    result = ""

    if st.button("Predict Fraud"):
        try:
            # Decode raw labels
            raw_category = category_options[selected_category]
            raw_gender = gender_options[selected_gender]
            raw_city = city_options[selected_city]
            raw_state = state_options[selected_state]

            # Encode categorical fields
            encoded_category = le_category.transform([raw_category])[0]
            encoded_gender = le_gender.transform([raw_gender])[0]
            encoded_city = le_city.transform([raw_city])[0]
            encoded_state = le_state.transform([raw_state])[0]

            # Collect all features
            input_data = [
                encoded_category,
                float(amt),
                encoded_gender,
                encoded_city,
                encoded_state,
                float(lat),
                float(long),
                float(city_pop)
            ]

            result = fraud_prediction(input_data)
        except Exception as e:
            result = f"⚠️ Error: {e}"

    st.success(result)

if __name__ == '__main__':
    main()
