from fastapi import FastAPI
import uvicorn
import os
import logging

from helpers.labeler import Labeler
from helpers.probe_row import ProbeRow
from helpers.sniffer_classifier_context import SnifferClassifierContext

app = FastAPI()


logger = logging.getLogger("antisniff-main")
@app.post("/predict")
def predict(row: ProbeRow):
    logger.info(f"""arguments:\n
            avg: {row.rtt_avg}\n
            mean: {row.rtt_median},\n
            flood: {row.flood_flag},\n
            max_diff: {row.max_diff},\n
            device: {row.device},\n
            ip: {row.ip}""")
    if row.flood_flag == 0:
        app.state.LABELER.label_machine(row)
    label = app.state.LABELER.get_label(row.device, row.ip)
    if label is None:
        return { "sniffing": -1.0, "label": "unknown"}

    if label == "laptop":
        model: SnifferClassifierContext = app.state.MODEL_LAPTOP
    else:
        model: SnifferClassifierContext = app.state.MODEL_PC

    probability  = model.classify(row)
    logger.info(f"res:\n {probability}\n")
    return { "sniffing": probability, "computer_type": label}

def main():
    flood_only = os.getenv("ANTISNIFFER_FLOOD_ONLY", True)
    # flood_only = False
    app.state.MODEL_LAPTOP = SnifferClassifierContext(flood_only, "laptop_model")
    app.state.MODEL_PC = SnifferClassifierContext(flood_only, "pc_model")
    app.state.LABELER = Labeler()
    uvicorn.run(app, host="0.0.0.0", port = 8001)

if __name__ == "__main__":
    main()
