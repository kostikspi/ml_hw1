from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import uvicorn


# Модели данных
class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


def transform_mileage(mileage):
    try:
        if pd.isna(mileage):
            return np.nan
        return float(mileage.split()[0])
    except:
        return 0


def transform_torque(row):
    try:
        torque_match = re.search(r"([\d.]+)\s?(Nm|kgm)", row, re.IGNORECASE)
        rpm_match = re.search(r"(\d+)[-–]?(\d+)?\s*rpm", row.replace(",", ""), re.IGNORECASE)

        if torque_match:
            torque_value = float(torque_match.group(1))
            if torque_match.group(2).lower() == "kgm":
                torque_value *= 9.80665
        else:
            torque_value = np.nan

        if rpm_match:
            max_rpm = float(rpm_match.group(2)) if rpm_match.group(2) else float(rpm_match.group(1))
        else:
            max_rpm = np.nan

        return pd.Series([torque_value, max_rpm])
    except:
        return pd.Series([0, 0])


class Items(BaseModel):
    objects: List[Item]


class InferenceModel:

    def __init__(self):
        with (open("model_ridge.pkl", "rb") as model_file,
              open("scaler.pkl", "rb") as scaler_file,
              open("encoder.pkl", "rb") as encoder_file):
            self.model = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.encoder = pickle.load(encoder_file)

    def predict(self, X):
        input_data = self.preprocess_data(X)
        output = self.model.predict(input_data)
        print('Done predicting')
        return output

    def preprocess_data(self, df: pd.DataFrame):
        df['mileage'] = df['mileage'].apply(transform_mileage)
        df['engine'] = df['engine'].apply(transform_mileage)
        df['max_power'] = df['max_power'].apply(transform_mileage)

        df[['torque', 'max_torque_rpm']] = df["torque"].apply(transform_torque)

        df['brand'] = df['name'].apply(lambda x: x.split(' ')[0])
        df.drop(['name'], axis=1, inplace=True)

        print('Done basic preprocess')
        categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'seats']
        encoded_features = self.encoder.transform(df[categorical_columns])
        df = df.drop(categorical_columns, axis=1)
        print('Done encoding preprocess')
        feature_names = self.encoder.get_feature_names_out(categorical_columns)
        df[feature_names] = encoded_features
        print('Done encoding preprocess and append')
        return df


app = FastAPI()
model = InferenceModel()


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    try:
        features = pd.DataFrame([item.dict()])
        prediction = model.predict(features)
        return prediction[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    try:
        features = pd.DataFrame([obj.dict() for obj in items.objects])
        predictions = model.predict(features)
        return predictions.tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=True)
