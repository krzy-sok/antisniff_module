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
    logger.info(f"""arguments:
            avg: {row.rtt_avg}
            mean: {row.rtt_median},
            flood: {row.flood_flag},
            max_diff: {row.max_diff},
            device: {row.device},
            ip: {row.ip}""")
    if row.flood_flag == 0:
        app.state.LABELER.label_machine(row)
    label = app.state.LABELER.get_label(row.device, row.ip)
    if label is None:
        return { "sniffing": -1.0, "computer_type": "unknown"}

    if label == "laptop":
        model: SnifferClassifierContext = app.state.MODEL_LAPTOP
    else:
        model: SnifferClassifierContext = app.state.MODEL_PC

    probability  = model.classify(row)
    logger.info(f"sniffing: {probability}, computer_type: {label}")
    return { "sniffing": probability, "computer_type": label}

def main():
    flood_only = os.getenv("ANTISNIFFER_FLOOD_ONLY", True)
    app.state.MODEL_LAPTOP = SnifferClassifierContext(flood_only, "laptop_model")
    app.state.MODEL_PC = SnifferClassifierContext(flood_only, "pc_model")
    app.state.LABELER = Labeler()
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | "
                           "%(module)s:%(funcName)s:%(lineno)d - %(message)s\n")
    uvicorn.run(app, host="0.0.0.0", port = 8001,  access_log=False,)

if __name__ == "__main__":
    main()
