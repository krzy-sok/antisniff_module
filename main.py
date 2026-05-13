from fastapi import FastAPI
import numpy as np
from helpers.load_model import load_model
from random import random
import uvicorn
from onnxruntime import InferenceSession

from helpers.labeler import Labeler
from helpers.probe_row import ProbeRow

app = FastAPI()

@app.post("/predict")
def predict(row: ProbeRow):
    print(f"""arguments:\n
            avg: {row.rtt_avg}\n
            mean: {row.rtt_median},\n
            flood: {row.flood_flag},\n
            max_diff: {row.max_diff},\n
            device: {row.device},\n
            ip: {row.ip}""")
    if row.flood_flag == 0:
        app.state.LABELER.label_machine(row)
    label = app.state.LABELER.get_label(row.device, row.ip)

    if label == "laptop":
        model: InferenceSession = app.state.MODEL_LAPTOP
    else:
        model: InferenceSession = app.state.MODEL_PC

    input = np.array([row.rtt_avg, row.rtt_median, float(row.flood_flag), row.max_diff])
    input = input.astype("float32")
    input = input.reshape(1,-1)
    res = model.run(["output_probability"], {"input": input})
    proba = res[0]
    print(f"res:\n {res}\n")
    return { "sniffing": proba[0][1], "label": label}

def main():
    app.state.MODEL_LAPTOP = load_model("laptop_model")
    app.state.MODEL_PC = load_model("pc_model")
    app.state.LABELER = Labeler()
    uvicorn.run(app, host="0.0.0.0", port = 8001)

if __name__ == "__main__":
    main()
