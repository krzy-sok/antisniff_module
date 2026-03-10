from fastapi import FastAPI
from random import random
import uvicorn

app = FastAPI()

@app.get("/predict")
def predict():
    return random()

def main():
    uvicorn.run(app, host="0.0.0.0", port = 8001)

if __name__ == "__main__":
    main()
