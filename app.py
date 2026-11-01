import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.train_model import (
    load_or_train_model, FEATURE_COLUMNS, load_data,
    make_preprocessor, make_models, prepare_train_test
)

st.set_page_config(page_title="Industrial Predictive Maintenance", page_icon="🏭", layout="wide")
st.title("🏭 Industrial Predictive Maintenance")
st.write("Predict machine failure risk and understand which operating measurements influence the prediction.")
st.caption("Educational project using the UCI AI4I 2020 Predictive Maintenance dataset.")

@st.cache_resource
def get_model_bundle():
    return load_or_train_model()

try:
    bundle = get_model_bundle()
    model = bundle["model"]
    metrics = bundle["metrics"]
except Exception as exc:
    st.error("Could not prepare the model. Check your internet connection and app logs.")
    st.exception(exc)
    st.stop()

tab1, tab2 = st.tabs(["Failure prediction", "Model performance & SHAP"])

with tab1:
    st.subheader("Enter machine operating measurements")
    with st.form("prediction_form"):
        machine_type = st.selectbox("Machine type", ["L", "M", "H"])
        left, right = st.columns(2)
        with left:
            air = st.number_input("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
            speed = st.number_input("Rotational speed (rpm)", 1000, 3000, 1500, 10)
            wear = st.number_input("Tool wear (minutes)", 0, 300, 100, 1)
        with right:
            process = st.number_input("Process temperature (K)", 300.0, 315.0, 310.0, 0.1)
            torque = st.number_input("Torque (Nm)", 3.0, 80.0, 40.0, 0.5)
        submitted = st.form_submit_button("Predict failure risk")

    if submitted:
        row = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear
        }], columns=FEATURE_COLUMNS)
        probability = float(model.predict_proba(row)[0][1])
        st.metric("Estimated failure probability", f"{probability:.1%}")
        if probability >= 0.5:
            st.error("Higher predicted risk — inspect the machine and operating conditions.")
        else:
            st.success("Lower predicted risk — continue normal monitoring.")
        st.caption("The displayed probability is a model estimate, not a safety guarantee.")

with tab2:
    st.subheader("Model comparison")
    st.json(metrics["model_comparison"])
    st.write(f"**Selected model:** {metrics['selected_model']}")
    st.info("The dataset is imbalanced. Precision, recall and F1-score are more useful than accuracy alone.")

    st.subheader("SHAP explainability")
    st.write(
        "SHAP estimates how much each input feature contributes to a model prediction. "
        "The global chart summarizes feature influence across test examples."
    )
    if st.button("Generate SHAP explanations"):
        with st.spinner("Calculating SHAP values..."):
            X_train, X_test, y_train, y_test = prepare_train_test()
            preprocessor = model.named_steps["preprocess"]
            classifier = model.named_steps["model"]
            transformed_test = preprocessor.transform(X_test)
            feature_names = preprocessor.get_feature_names_out()
            # TreeExplainer supports the Random Forest and XGBoost models used here.
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_test)
            if isinstance(shap_values, list):
                positive_values = shap_values[1]
            elif getattr(shap_values, "ndim", 0) == 3:
                positive_values = shap_values[:, :, 1]
            else:
                positive_values = shap_values
            st.markdown("**Global feature importance**")
            fig, ax = plt.subplots()
            shap.summary_plot(
                positive_values, transformed_test,
                feature_names=feature_names, plot_type="bar", show=False
            )
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close("all")

            st.markdown("**How features influence one test prediction**")
            expected = explainer.expected_value
            if hasattr(expected, "__len__"):
                base_value = expected[1]
            else:
                base_value = expected
            explanation = shap.Explanation(
                values=positive_values[0],
                base_values=base_value,
                data=transformed_test[0].toarray().ravel() if hasattr(transformed_test[0], "toarray") else transformed_test[0],
                feature_names=feature_names
            )
            fig = plt.figure()
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close("all")

st.divider()
st.caption("Not intended for real industrial safety or maintenance decisions without validation on real factory data.")
