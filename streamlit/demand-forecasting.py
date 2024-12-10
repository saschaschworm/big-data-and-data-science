import datetime
import itertools
import numpy as np
import pandas as pd
import sklearn
import skops.io as sio
import streamlit as st
from urllib3 import PoolManager

# Configuration for Scikit-Learn
sklearn.set_config(transform_output="pandas")

# Configuration for Streamlit
st.set_page_config(page_title="Demand Forecasting", layout="wide")

# Helper Utilities
https = PoolManager()
test = lambda x: x.copy().ffill()

markflgs = {"LOW": "Low Marketing Activity", "MEDIUM": "Medium Marketing Activity", "HIGH": "High Marketing Activity"}
promflgs = {"NONE": "No Promotion", "BOGO": "Buy One Get One Free", "DISCOUNT": "Discounted Price"}
combinations = list(itertools.product(markflgs.keys(), promflgs.keys()))

cafflgs = {"NO": "No Customer Activity", "YES": "Kundenaktivität"}
features = ["TDATE", "DEMAND", "SEASON", "HOLIDAYS", "MARKETING", "PROMOTION"]
features = [*features, "CAF", "CCI", "NTI", "TEMPERATURE", "PRECIPITATION", "NOISE", "RPRC"]


@st.cache_resource
def load_model():
    """Load the demand forecasting model from the repository."""
    url = "https://github.com/saschaschworm/big-data-and-data-science/raw/refs/heads/master/models/demand-forecasting.skops"
    payload = https.request("GET", url).data
    unknown_types = sio.get_untrusted_types(data=payload)
    return sio.loads(payload, trusted=unknown_types)


@st.cache_data
def load_data():
    """Load the demand forecasting dataset from the repository."""
    url = "https://github.com/saschaschworm/big-data-and-data-science/raw/refs/heads/master/datasets/demand-forecasting-raw.feather"
    return pd.read_feather(url)


def toggle_simulation():
    if st.session_state.simulation:
        st.session_state.simulation = False
    else:
        st.session_state.simulation = True


data = load_data()
model = load_model()

if "simulation" not in st.session_state:
    st.session_state.simulation = False

st.title("Demand Forecasting for FreshMart")

oleft, ileft, iright, oright = st.columns(4)
with oleft:
    with st.container(border=True):
        st.metric(label="Last Training Date", value=data["ODATE"].dt.strftime("%d.%m.%Y").tail(1).values[0])


col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.header("Parameters for Forecasting")

        left, right = st.columns(2)
        with left:
            date = st.date_input("Forecast Date", value=datetime.date(2024, 8, 31), format="DD.MM.YYYY")
            demand = st.number_input("Demand Prior to the Forecast Date", step=1, value=300)
        with right:
            holidays = st.number_input("Number of Holidays after Forecast Date", step=1)
            rprc = st.number_input("Selling Price in Euro", step=0.01, value=2.19)

        left, right = st.columns(2)
        with left:
            cci = st.number_input("Consumer Confidence Index", step=0.1, value=-18.6)
        with right:
            nti = st.number_input("Nutritional Trends Index", step=1, value=132)

        left, right = st.columns(2)
        with left:
            precipitation = st.number_input(
                "Precipitation Probability in Percent",
                step=0.1,
                min_value=0.0,
                max_value=100.0,
                value=7.0,
                format="%.1f",
            )
        with right:
            temperature = st.number_input(
                "Average Daily Temperature in Degrees Celsius", step=0.1, value=27.7, format="%.1f"
            )

        left, right = st.columns(2)
        with left:
            promotion = st.selectbox(
                "Promotion",
                options=["NONE", "BOGO", "DISCOUNT"],
                format_func=lambda x: promflgs[x],
                disabled=st.session_state.simulation,
            )
        with right:
            marketing = st.selectbox(
                "Marketing",
                options=["LOW", "MEDIUM", "HIGH"],
                format_func=lambda x: markflgs[x],
                disabled=st.session_state.simulation,
            )

        caf = st.selectbox("Customer Activity Flag", options=["NO", "YES"], format_func=lambda x: cafflgs[x])

        left, right = st.columns(2)
        with left:
            submitted = st.button("Forecast Demand", type="primary", use_container_width=True)
        with right:
            simulation = st.toggle(
                "Recommend Optimal Marketing & Promotion Strategy",
                value=st.session_state.simulation,
                on_change=toggle_simulation,
            )


with col2:
    with st.container(border=True):
        st.header("Forecast Results")

        if submitted:
            labels = ["WINTER", "SPRING", "SUMMER", "FALL", "WINTER"]
            season = pd.cut([date.month], bins=[-999, 2, 5, 8, 11, 999], labels=labels, ordered=False)[0]

            results = []
            simulations = combinations if simulation else [(marketing, promotion)]
            for marketing, promotion in simulations:
                df = pd.DataFrame()
                df["TDATE"] = pd.to_datetime([date])
                df["DEMAND"] = demand
                df["SEASON"] = season
                df["HOLIDAYS"] = holidays
                df["MARKETING"] = marketing
                df["PROMOTION"] = promotion
                df["CAF"] = caf
                df["CCI"] = cci
                df["NTI"] = nti
                df["TEMPERATURE"] = temperature
                df["PRECIPITATION"] = precipitation / 100
                df["NOISE"] = data["NOISE"].mean()
                df["RPRC"] = rprc
                df = pd.concat([data[features], df], axis=0).reset_index(drop=True)

                forecast = pd.DataFrame(model.predict(df)).tail(1)
                results.append({"marketing": marketing, "promotion": promotion, "v": np.ceil(forecast.values[0][0])})

            df = pd.DataFrame(results)
            df.columns = ["Marketing", "Promotion", "Forecasted Demand"]
            df["Forecasted Demand"] = df["Forecasted Demand"].astype(int)

            st.dataframe(df, use_container_width=True)

            if simulation:
                best = df[df["Forecasted Demand"] == df["Forecasted Demand"].max()]
                st.success(
                    f"Based on the forecast, the optimal marketing strategy is a **{markflgs[best['Marketing'].values[0]].lower()}** and the optimal promotion strategy is **{promflgs[best['Promotion'].values[0]].lower()}**."
                )

        else:
            st.text("Please submit the form to forecast the demand.")
