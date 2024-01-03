import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np

if __name__ == "__main__":
    title = "Evaluation Metrics for Binary Classification"
    st.set_page_config(page_title=title, layout="wide", initial_sidebar_state="expanded")
    st.title(title)

    df = pd.DataFrame([
        {"Training Instance": 1, "Actual Label": True}, 
        {"Training Instance": 2, "Actual Label": False}, 
        {"Training Instance": 3, "Actual Label": True}, 
        {"Training Instance": 4, "Actual Label": False}, 
        {"Training Instance": 5, "Actual Label": True}, 
        {"Training Instance": 6, "Actual Label": True}, 
        {"Training Instance": 7, "Actual Label": False}, 
        {"Training Instance": 8, "Actual Label": True}, 
        {"Training Instance": 9, "Actual Label": False}, 
        {"Training Instance": 10, "Actual Label": False}
    ])
    df.set_index("Training Instance", inplace=True)

    columns = st.columns(2)
    with columns[0]:

        with st.container(border=True):
            st.write("Below you have the option of adjusting the classification threshold as well as changing the \"true\" labels of the data set. Depending on your adjustments, you will immediately see the effects on the predicted labels and on relevant evaluation metrics. This enables you to understand why, for example, recall is an important measure when false negatives are associated with high costs. The classification threshold is required because classifiers often return probabilities and these probabilities must then be assigned to a binary category depending on the classification threshold.")
            probability = st.slider("**Classification Threshold**", 0, 100, 50, 5, format="%i %%")
            df = st.data_editor(df.T, use_container_width=True)

        df = df.T.merge(pd.DataFrame([
            {"Training Instance": "1", "Predicted Probability": 95},
            {"Training Instance": "2", "Predicted Probability": 15},
            {"Training Instance": "3", "Predicted Probability": 75},
            {"Training Instance": "4", "Predicted Probability": 55},
            {"Training Instance": "5", "Predicted Probability": 65},
            {"Training Instance": "6", "Predicted Probability": 45},
            {"Training Instance": "7", "Predicted Probability": 35},
            {"Training Instance": "8", "Predicted Probability": 85},
            {"Training Instance": "9", "Predicted Probability": 5},
            {"Training Instance": "10", "Predicted Probability": 25},
        ]), on="Training Instance")
        df["Predicted Label"] = df["Predicted Probability"] >= probability
        df["Result"] = np.where((df["Actual Label"] == True) & (df["Predicted Label"] == True), "True Positive (TP)", 0)
        df["Result"] = np.where((df["Actual Label"] == True) & (df["Predicted Label"] == False), "False Negative (FN)", df["Result"])
        df["Result"] = np.where((df["Actual Label"] == False) & (df["Predicted Label"] == False), "True Negative (TN)", df["Result"])
        df["Result"] = np.where((df["Actual Label"] == False) & (df["Predicted Label"] == True), "False Positive (FP)", df["Result"])

        tp = len(df[df["Result"] == "True Positive (TP)"])
        tn = len(df[df["Result"] == "True Negative (TN)"])
        fp = len(df[df["Result"] == "False Positive (FP)"])
        fn = len(df[df["Result"] == "False Negative (FN)"])
        accuracy = (tp + tn) / len(df)

        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = None
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = None

        try:
            fscore = 2 * ((precision * recall) / (precision + recall))
        except (TypeError, ZeroDivisionError):
            fscore = None

        st.dataframe(df, use_container_width=True, hide_index=True, column_config={
            "Predicted Probability": st.column_config.NumberColumn(format="%.2f %%"),
        })
    
    with columns[1]:
        df["Actual Label"] = df["Actual Label"].map({True: "Actual Positive", False: "Actual Negative"})
        df["Predicted Label"] = df["Predicted Label"].map({True: "Predicted Positive", False: "Predicted Negative"})
        with st.container(border=True):
            fig = px.scatter(x=df["Training Instance"], y=df["Predicted Probability"], color=df["Actual Label"], symbol=df["Predicted Label"])
            fig.add_hline(y=probability)
            fig.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=1, range=[0.5, 10.5]))
            fig.update_layout(yaxis=dict(tickmode="linear", tick0=0, dtick=10, ticksuffix=" %", range=[0, 105]))
            fig.update_layout(xaxis_title="Training Instance", yaxis_title="Predicted Probability")
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None))
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            fig.update_traces(marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pd.DataFrame([
            {"Metric": "True Positives", "Value": f"{tp}", "Description": "A true positive is an outcome where the model correctly predicts the positive class."},
            {"Metric": "True Negatives", "Value": f"{tn}", "Description": "A true negative is an outcome where the model correctly predicts the negative class."},
            {"Metric": "False Positives", "Value": f"{fp}", "Description": "A false positive is an outcome where the model incorrectly predicts the positive class."},
            {"Metric": "False Negatives", "Value": f"{fn}", "Description": "A false negative is an outcome where the model incorrectly predicts the negative class."},
            {"Metric": "Accuracy", "Value": f"{accuracy * 100:.2f} %", "Description": "What fraction of predictions the model got right?"},
            {"Metric": "Precision/Positive Predictive Value", "Value": f"{precision * 100:.2f} %" if precision else "N/A", "Description": "What proportion of positive identifications was actually correct?"},
            {"Metric": "Recall/True Positive Rate", "Value": f"{recall * 100:.2f} %" if recall else "N/A", "Description": "What proportion of actual positives was identified correctly?"},
            {"Metric": "F1", "Value": f"{fscore * 100:.2f} %" if fscore else "N/A", "Description": "How good can the model effectively identify positive cases while minimizing false positives and false negatives?"},
        ]), use_container_width=True, hide_index=True)
