#%% imports
SEED = 123

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import numpy as np
np.random.seed(SEED)


import torch
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#%% Define needed dataset class
class NasaDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#%% Read in, normalize, create datset/data loader
raw_d = pd.read_csv(r"C:\Users\T480s\OneDrive\Documents\aKTH\KTH Second Year\QNN\DD2368_assignments\final_project\nasa_data\neo.csv")

feats = [ # we don't care about id, name, sentry_object, orbiting_body
    "est_diameter_min",
    "est_diameter_max",
    "relative_velocity",
    "miss_distance",
    "absolute_magnitude",
]

X = (raw_d[feats].values) # shape (num_rows, 5)
y = (raw_d["hazardous"].values) # shape (num_rows,)

X_tr, X_temp, y_tr, y_temp = train_test_split(X, 
                                y, 
                                test_size=0.3, 
                                random_state=SEED, 
                                stratify=y
                                )
X_val, X_te, y_val, y_te = train_test_split(X_temp, 
                                y_temp, 
                                test_size=0.5, 
                                random_state=SEED,
                                stratify=y_temp
                                )

scaler = StandardScaler()
X_tr = scaler.fit_transform(X=X_tr)
X_val = scaler.transform(X=X_val)
X_te = scaler.transform(X=X_te)

# convert to torch tensors, sending to device here.
# could cause problems later, if getting error check this
X_tr  = torch.tensor(X_tr, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
X_te  = torch.tensor(X_te, dtype=torch.float32).to(device)

y_tr  = torch.tensor(y_tr, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
y_te  = torch.tensor(y_te, dtype=torch.float32).to(device)

train_ds = NasaDataset(X_tr, y_tr)
val_ds   = NasaDataset(X_val, y_val)
test_ds  = NasaDataset(X_te, y_te)

train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False
)

test_loader = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False
)
print (X_tr)

# sanity checks
assert X.shape[0] == X_tr.shape[0] + X_val.shape[0] + X_te.shape[0]

print (X_tr.shape[0], X_val.shape[0], X_te.shape[0])

# for features, labels in train_loader:
#     print(features.shape, labels)

# %%
