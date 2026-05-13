import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import requests


def send_requests(x_data):
    y_pred_label = []
    y_pred_sniff = []

    for i, x_row in x_data.iterrows():
        json= {
            "rtt_avg": x_row["rtt_avg"],
            "rtt_median": x_row["rtt_median"],
            "flood_flag" : x_row["flood_flag"],
            "max_diff": x_row["max_diff"],
            "device": "a",
            "ip": x_row["ip"]
        }
        res = requests.post("http://localhost:8001/predict", json= json)
        data = res.json()
        y_pred_label.append(data["label"])
        y_pred_sniff.append(data["sniffing"])
    return y_pred_label, y_pred_sniff


def test_clsssify_pc():
    df =  pd.read_csv("test/csvs/12.05-pc-labeled.csv")
    df["ip"] = "192.168.0.102"
    x_data = df[["flood_flag", "rtt_avg", "rtt_median", "is_sniffing", "max_diff", "ip"]]
    y_sniff = df["is_sniffing"].astype(int)

    _, y_pred_sniff = send_requests(x_data)

    y_pred_sniff = [int(round(proba)) for proba in y_pred_sniff]
    report_dict = classification_report(y_sniff, y_pred_sniff, output_dict=True)
    assert(report_dict["0"]["support"] >= 200)
    assert(report_dict["1"]["support"] >= 200)
    assert(report_dict["0"]["precision"] >= 0.9)
    assert(report_dict["0"]["recall"] >= 0.9)
    assert(report_dict["1"]["precision"] >= 0.9)
    assert(report_dict["1"]["recall"] >= 0.9)


def test_classify_laptop():
    df = pd.read_csv("test/csvs/12.05-test-lap-labeled.csv")
    df["ip"] = "192.168.0.102"
    x_data = df[["flood_flag", "rtt_avg", "rtt_median", "is_sniffing", "max_diff", "ip"]]
    # y_label = df["computer_type"]
    y_sniff = df["is_sniffing"].astype(int)

    _, y_pred_sniff = send_requests(x_data)

    y_pred_sniff = [int(round(proba)) for proba in y_pred_sniff]
    report_dict = classification_report(y_sniff, y_pred_sniff, output_dict=True)
    assert(report_dict["0"]["support"] >= 200)
    assert(report_dict["1"]["support"] >= 200)
    assert(report_dict["0"]["precision"] >= 0.9)
    assert(report_dict["0"]["recall"] >= 0.9)
    assert(report_dict["1"]["precision"] >= 0.9)
    assert(report_dict["1"]["recall"] >= 0.9)


def test_classify_computer_type():
    df_pc = pd.read_csv("test/csvs/12.05-pc-labeled.csv")
    df_lap = pd.read_csv("test/csvs/12.05-test-lap-labeled.csv")

    df_pc["ip"] = "192.168.0.101"
    df_lap["ip"] = "192.168.0.102"

    df = pd.concat([df_pc, df_lap])
    x_data = df[["flood_flag", "rtt_avg", "rtt_median", "is_sniffing", "max_diff", "ip"]]
    y_label = df["computer_type"]

    y_pred_label, _ = send_requests(x_data)
    report_dict = classification_report(y_label, y_pred_label, output_dict=True)
    assert(report_dict["laptop"]["support"] >= 400)
    assert(report_dict["pc"]["support"] >= 400)
    assert(report_dict["laptop"]["precision"] >= 0.95)
    assert(report_dict["laptop"]["recall"] >= 0.95)
    assert(report_dict["pc"]["precision"] >= 0.95)
    assert(report_dict["pc"]["recall"] >= 0.95)


def test_classify_all():
    df_pc = pd.read_csv("test/csvs/12.05-pc-labeled.csv")
    df_lap = pd.read_csv("test/csvs/12.05-test-lap-labeled.csv")

    df_pc["ip"] = "192.168.0.101"
    df_lap["ip"] = "192.168.0.102"

    df = pd.concat([df_pc, df_lap])

    x_data = df[["flood_flag", "rtt_avg", "rtt_median", "is_sniffing", "max_diff", "ip"]]
    y_label = df["computer_type"]
    y_sniff = df["is_sniffing"].astype(int)

    y_pred_label, y_pred_sniff = send_requests(x_data)

    report_dict_labels = classification_report(y_label, y_pred_label, output_dict=True)
    assert(report_dict_labels["laptop"]["support"] >= 400)
    assert(report_dict_labels["pc"]["support"] >= 400)
    assert(report_dict_labels["laptop"]["precision"] >= 0.95)
    assert(report_dict_labels["laptop"]["recall"] >= 0.95)
    assert(report_dict_labels["pc"]["precision"] >= 0.95)
    assert(report_dict_labels["pc"]["recall"] >= 0.95)

    y_pred_sniff = [int(round(proba)) for proba in y_pred_sniff]
    report_dict = classification_report(y_sniff, y_pred_sniff, output_dict=True)
    assert(report_dict["0"]["support"] >= 400)
    assert(report_dict["1"]["support"] >= 400)
    assert(report_dict["0"]["precision"] >= 0.9)
    assert(report_dict["0"]["recall"] >= 0.9)
    assert(report_dict["1"]["precision"] >= 0.9)
    assert(report_dict["1"]["recall"] >= 0.9)



# print("\n---- labeling ----")
# print(classification_report(y_label, y_pred_label))
# print(confusion_matrix(y_label, y_pred_label))
# # breakpoint()
# y_pred_sniff = [int(round(proba)) for proba in y_pred_sniff]
# print("\n---- sniffing ----")
# print(classification_report(y_sniff, y_pred_sniff))
# print(confusion_matrix(y_sniff, y_pred_sniff))