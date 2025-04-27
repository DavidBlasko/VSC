import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# get path to training .csv file
csv_path = sys.argv[1]

# load training data
df = pd.read_csv(csv_path)

# separate features (X) and labels (y)
X = df.drop("class", axis=1).values
y = df["class"].map({"QqQqQ": 0, "MmMmM": 1}).values # encode labels as 0 and 1

# scale features using StandardScaler (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split data into training and validation sets (80/20 split)
X_train, X_validate, y_train, y_validate = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# build neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)), # first hidden layer
    BatchNormalization(),
    Dropout(0.3), # regularization to prevent overfitting
    
    Dense(64, activation='relu'), # second hidden layer
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'), # third hidden layer
    BatchNormalization(),
    
    Dense(1, activation='sigmoid') # output layer for binary classification
])

# model compilation with Adam optimizer and binary crossentropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validate, y_validate))

# save model and scaler params
save_model(model, "model.h5")
np.savez("scaler.npz", mean=scaler.mean_, scale=scaler.scale_)