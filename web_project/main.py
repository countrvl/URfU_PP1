from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
import os
from io import StringIO
import base64

app = FastAPI()

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
PREDICTION_FILE = "test_with_predictions.csv"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Стартовая страница с формой для загрузки файла"""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload_csv/")
async def upload_csv(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    try:
        data = pd.read_csv(StringIO(contents.decode("utf-8")))

        if "id" in data.columns:
            ids = data["id"]
            data = data.drop(columns=["id"])
        else:
            ids = range(1, len(data) + 1)

        numerical_columns = [
            "Previous qualification (grade)", "Admission grade", "Age at enrollment",
            "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
            "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
            "Unemployment rate", "Inflation rate", "GDP"
        ]

        data[numerical_columns] = scaler.transform(data[numerical_columns])

        predictions = model.predict(data)
        data["Prediction"] = predictions
        data.to_csv(PREDICTION_FILE, index=False)

        plt.figure(figsize=(6, 4))
        plt.hist(predictions, bins=2, color='skyblue', alpha=0.7, edgecolor='black')
        plt.xticks([0, 1], ['Неуспешный', 'Успешный'])
        plt.title("Распределение предсказаний")
        plt.xlabel("Категория")
        plt.ylabel("Количество")
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        preview_data = data.head().to_html(classes="table table-striped", index=False)

        return templates.TemplateResponse("results.html", {
            "request": request,
            "graph_url": f"data:image/png;base64,{graph_url}",
            "preview_data": preview_data
        })
    except Exception as e:
        return {"error": f"Ошибка обработки файла: {str(e)}"}

@app.get("/download/")
async def download_file():
    return FileResponse(path=PREDICTION_FILE, filename="test_with_predictions.csv", media_type="text/csv")
