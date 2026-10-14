from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

BASE = Path(__file__).resolve().parent


# The model was saved from the notebook: these definitions must match it, since the saved file refers to them by name
def cleaning(df):
    df = df.copy()
    df["weight"] = df["weight"].abs()
    return df


class MyStatefulTransformer(BaseEstimator, TransformerMixin):
    """Fill missing values with statistics learned on the training data."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.median_ = X["weight"].median()
        self.median_marketindex_ = X["market_index"].median()
        return self

    def transform(self, X):
        check_is_fitted(self, "median_")
        X = pd.DataFrame(X).copy()
        X["weight"] = X["weight"].fillna(self.median_)
        date_median = X.groupby("date")["market_index"].transform("median")
        X["market_index"] = (X["market_index"]
                             .fillna(date_median)
                             .fillna(self.median_marketindex_))
        return X


def add_feats(df, which=()):
    df = df.copy()
    if "dow" in which:
        df["dow"] = df["date"].dt.dayofweek
    if "interact" in which:
        df["dist_x_market"] = df["distance"] * df["market_index"]
        df["dist_x_quote"] = df["distance"] * df["quote_signal"]
    if "direction" in which:
        df["d_lat"] = df["delivery_lat"] - df["pickup_lat"]
        df["d_lon"] = df["delivery_lon"] - df["pickup_lon"]
    return df


class RpmTarget(BaseEstimator, RegressorMixin):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, y / X["distance"])
        return self

    def predict(self, X):
        return self.regressor_.predict(X) * X["distance"].to_numpy()


REQUIRED = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
            "distance", "weight", "market_index", "equipment", "date"]


@st.cache_resource
def load_model():
    return joblib.load(BASE / "freight_rate_model.joblib")


@st.cache_data
def load_reference():
    data = pd.read_csv(BASE / "train-test.csv")
    pickup = data.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    delivery = data.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
    return pickup, delivery, sorted(data["equipment"].unique())


def predict(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return load_model().predict(df)


st.set_page_config(page_title="Freight rate prediction")
st.title("Freight rate prediction")
st.caption("CatBoost model trained on the labelled loads (Jan-Oct). Predicted rates are in USD.")

pickup_cities, delivery_cities, equipments = load_reference()
mode = st.radio("Input", ["One load", "CSV file"], horizontal=True)

if mode == "One load":
    col1, col2 = st.columns(2)
    pickup = col1.selectbox("Pickup city", pickup_cities.index)
    delivery = col2.selectbox("Delivery city", delivery_cities.index)
    distance = col1.number_input("Distance (miles)", min_value=1, value=360)
    weight = col2.number_input("Weight (lb)", min_value=0, value=32000)
    equipment = col1.selectbox("Equipment", equipments)
    date = col2.date_input("Pickup date", value=pd.Timestamp("2025-12-01"))
    market_index = st.number_input("market_index", min_value=0.0, value=1.0, step=0.01,
                                   help="Market level of the day, as in the data files.")

    if st.button("Predict"):
        row = pd.DataFrame([{
            **pickup_cities.loc[pickup].to_dict(),
            **delivery_cities.loc[delivery].to_dict(),
            "distance": distance, "weight": weight, "market_index": market_index,
            "equipment": equipment, "date": date,
        }])
        rate = float(predict(row)[0])
        st.metric("Predicted rate", f"${rate:,.0f}")
        st.caption(f"{rate / distance:.2f} USD per mile")

else:
    st.write("Required columns: " + ", ".join(REQUIRED))
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        missing = [c for c in REQUIRED if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            result = data.copy()
            result["predicted_rate"] = np.round(predict(data), 2)
            st.write(f"{len(result):,} loads")
            st.dataframe(result.head(100))
            st.write("Predicted rate summary")
            st.dataframe(result["predicted_rate"].describe().round(1).to_frame().T)
            st.download_button("Download predictions", result.to_csv(index=False).encode("utf-8"),
                               file_name="predictions.csv", mime="text/csv")
