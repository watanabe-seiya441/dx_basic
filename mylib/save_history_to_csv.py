import os
import csv

def save_history_to_csv(history, save_dir):
    filename = os.path.join(save_dir, "history.csv")
    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)

        for row in history:
            csvwriter.writerow(row)
    print(f"配列が {filename} に保存されました。")