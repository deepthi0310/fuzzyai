import pandas as pd
import numpy as np
import random

data = []

for _ in range(300):  # 300 samples
    ph = round(random.uniform(0, 14), 2)
    turbidity = round(random.uniform(0, 100), 2)
    temp = round(random.uniform(0, 50), 2)

    # A rough logic for deciding the dosage
    if ph < 4 or turbidity > 60 or temp > 35:
        dosage = random.randint(70, 100)
    elif 4 <= ph <= 10 and 20 <= turbidity <= 60 and 15 <= temp <= 35:
        dosage = random.randint(40, 70)
    else:
        dosage = random.randint(0, 40)

    data.append([ph, turbidity, temp, dosage])

df = pd.DataFrame(data, columns=["ph", "turbidity", "temperature", "dosage"])
df.to_csv("dosage_dataset.csv", index=False)
