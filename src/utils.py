from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_reg(df):
    reg = LinearRegression()
    reg.fit(df.iloc[:,[0]],df.iloc[:,[1]])
    x = df.iloc[:,[0]]
    y = reg.predict(x)
    plt.plot(x,y,lw=1.6)



def simulate_data(seed=99):
    np.random.seed(seed)
    
    severity_levels = [
        "Mild",
        "Moderate",
        "Severe",
    ]
    n_per_group = 100
    data = []

    for i, sev in enumerate(severity_levels):
        
        # dosage increases with severity
        dosage = np.random.normal(loc=i+6, scale=1.2, size=n_per_group)

        # higher dosage -> faster recovery
        recovery_time = np.random.normal(7 + 2.5*i, 1.1, n_per_group) - 0.35*dosage

        # ensure reasonable bounds
        dosage = np.clip(dosage, 0.2, 10)
        recovery_time = np.clip(recovery_time, 0.0, None)
        
        for x, y in zip(dosage, recovery_time):
            data.append([x, y, sev])

    return pd.DataFrame(data, columns=["dosage", "recovery_time", "severity"])