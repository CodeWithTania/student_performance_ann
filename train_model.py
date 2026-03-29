# Step 1: Libraries import
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 2: Dataset load
data = pd.read_csv("dataset/student_data.csv")

# Step 3: Input aur Output split
X = data.drop("result", axis=1)
y = data["result"]

# Step 4: Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Data scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: ANN model build
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(4,)))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Step 7: Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 8: Train model
model.fit(X_train, y_train, epochs=50, batch_size=4)

# Step 9: Save model
model.save("model/ann_model.h5")

print("✅ ANN Model trained and saved successfully")
