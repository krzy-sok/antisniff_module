from fastapi import FastAPI
import numpy as np
from load_model import load_model
from random import random
import uvicorn

from probe_row import ProbeRow

app = FastAPI()

@app.post("/predict")
def predict(row: ProbeRow):
    model = app.state.MODEL
    input = np.array([row.rtt_avg, row.rtt_median, float(row.flood_flag)])
    input = input.astype("float32")
    input = input.reshape(1,-1)
    res = model.run(["output_probability"], {"input": input})
    proba = res[0]
    return proba[0][1]

def main():
    app.state.MODEL = load_model()
    uvicorn.run(app, host="0.0.0.0", port = 8001)

if __name__ == "__main__":
    main()
