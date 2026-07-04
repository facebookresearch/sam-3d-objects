import os
import pandas as pd


class TrainingLogger:
    def __init__(self):
        self.rows = []

    def add(self, row):
        self.rows.append(row)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(self.rows)
        df.to_csv(path, index=False)