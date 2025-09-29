import pandas as pd
import numpy as np

# Sample size
n_samples = 1000

# Simulating data
np.random.seed(42)

data = {
    'email_length': np.random.normal(100, 20, n_samples).astype(int),
    'contains_free': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
    'time_of_day': np.random.choice(['morning', 'afternoon', 'evening', 'night'], n_samples),
    'label': np.random.choice(['spam', 'ham'], n_samples, p=[0.4, 0.6])
}

df = pd.DataFrame(data)

#(Hack to introduce some relationships, rather than complete independence...)
for index, row in df.iterrows():
    prob = min(1, .7 *row["contains_free"] + .007*row["email_length"]+.1)
    df.at[index, 'label'] = np.random.choice(['spam', 'ham'], p=[prob, 1-prob])

# Saving the dataset
df.to_csv('simulated_email_dataset.csv', index=False)

    