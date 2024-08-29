#!/usr/bin/env python
# coding: utf-8

### Nayan Patel DL

# In[ ]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, GRU
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.feature_selection import RFE
from sklearn.svm import SVC


def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df, columns_to_drop=None):
    """Preprocesses the data by dropping specified columns, handling missing values, and scaling."""
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    df = df.dropna()  # Drop rows with missing values
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return scaled_df

def calculate_vif(df):
    """Calculates Variance Inflation Factor (VIF) for each feature in the DataFrame."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

def recursive_feature_elimination(X, y):
    """Performs Recursive Feature Elimination (RFE) using a Support Vector Classifier."""
    svc = SVC(kernel="linear", random_state=42)
    rfe = RFE(estimator=svc, n_features_to_select=10)
    rfe.fit(X, y)
    return rfe

def custom_class_weight(y):
    """Generates custom class weights based on class distribution."""
    c0, c1 = np.bincount(y)
    w0 = (1 / c0) * len(y) / 2
    w1 = (1 / c1) * len(y) / 2
    return {0: w0, 1: w1}

def apply_scaling(X_train, X_test):
    """Applies MinMax scaling to the training and test data."""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def build_model(hp):
    tf.keras.backend.clear_session()
    
    """ Build the base Model (LSTM Layer-1) with Hyperparameter Tuning"""
    # Instantiate the model
    model = Sequential()
    
    # Tune the number of units for the single LSTM layer
    hp_units = hp.Int('units', min_value=4, max_value=32, step=4)
    
    # Tune the dropout rate
    hp_dropout = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    
    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    # Add the single LSTM layer
    model.add(LSTM(units=hp_units, input_shape=(lookback, X_train_scaled.shape[1]), 
                   activation='relu', return_sequences=False))
    
    # Add Dropout layer
    model.add(Dropout(hp_dropout))
    
    # Add output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate, epsilon=1e-08, decay=0.0),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    return model


def build_model(hp):
    tf.keras.backend.clear_session()
    """ Build the 2-layer LSTM Model with hyperparametr tuning"""
    # Instantiate the model
    model = Sequential()
    
    # Tune the number of units for the first LSTM layer
    hp_units1 = hp.Int('units1', min_value=4, max_value=32, step=4)
    
    # Tune the dropout rate for the first LSTM layer
    hp_dropout1 = hp.Float('dropout_rate1', min_value=0.0, max_value=0.5, step=0.1)
    
    # Add the first LSTM layer
    model.add(LSTM(units=hp_units1, input_shape=(lookback, X_train_scaled.shape[1]), 
                   activation='relu', return_sequences=True))
    
    # Add Dropout after the first LSTM layer
    model.add(Dropout(hp_dropout1))
    
    # Tune the number of units for the second LSTM layer
    hp_units2 = hp.Int('units2', min_value=4, max_value=32, step=4)
    
    # Tune the dropout rate for the second LSTM layer
    hp_dropout2 = hp.Float('dropout_rate2', min_value=0.0, max_value=0.5, step=0.1)
    
    # Add the second LSTM layer
    model.add(LSTM(units=hp_units2, activation='relu', return_sequences=False))
    
    # Add Dropout after the second LSTM layer
    model.add(Dropout(hp_dropout2))
    
    # Add output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]), 
                                 epsilon=1e-08, decay=0.0),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    return model



def build_model(hp):
    tf.keras.backend.clear_session()
    
    model = Sequential()
    
    """Build the 5-Layer LSTM Model with Hyperparameter Tuning"""
    
    # First LSTM Layer
    hp_units1 = hp.Int('units1', min_value=4, max_value=64, step=4)
    hp_dropout1 = hp.Float('dropout_rate1', min_value=0.0, max_value=0.5, step=0.1)
    model.add(LSTM(units=hp_units1, input_shape=(lookback, X_train_scaled.shape[1]), activation='relu', return_sequences=True))
    model.add(Dropout(hp_dropout1))
    
    # Second to Fifth LSTM Layers
    for i in range(2, 6):
        hp_units = hp.Int(f'units{i}', min_value=4, max_value=64, step=4)
        hp_dropout = hp.Float(f'dropout_rate{i}', min_value=0.0, max_value=0.5, step=0.1)
        model.add(LSTM(units=hp_units, activation='relu', return_sequences=True if i < 5 else False))
        model.add(Dropout(hp_dropout))
    
    # Output Layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the Model
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]), epsilon=1e-08, decay=0.0),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    return model



def build_model(hp):
    tf.keras.backend.clear_session()
    
    """Build the GRU Model with Hyperparameter Tuning"""
    
    # Instantiate the model
    model = Sequential()
    
    # Tune the number of units for the first GRU layer
    hp_units1 = hp.Int('units1', min_value=4, max_value=64, step=4)
    # Tune the dropout rate for the first GRU layer
    hp_dropout1 = hp.Float('dropout_rate1', min_value=0.0, max_value=0.5, step=0.1)
    
    # Add the first GRU layer
    model.add(GRU(units=hp_units1, input_shape=(lookback, X_train_scaled.shape[1]), 
                  activation='relu', return_sequences=True))
    # Add Dropout after the first GRU layer
    model.add(Dropout(hp_dropout1))
    
    # Add a second GRU layer
    hp_units2 = hp.Int('units2', min_value=4, max_value=64, step=4)
    hp_dropout2 = hp.Float('dropout_rate2', min_value=0.0, max_value=0.5, step=0.1)
    model.add(GRU(units=hp_units2, activation='relu', return_sequences=False))
    model.add(Dropout(hp_dropout2))
    
    # Add output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]), 
                                 epsilon=1e-08, decay=0.0),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    return model




def build_model(hp):
    tf.keras.backend.clear_session()
    
    """Step 1: Build the LSTM + CNN Model with Hyperparameter Tuning"""
    
    # Input Layer
    inputs = Input(shape=(lookback, X_train_scaled.shape[1]))
    
    # CNN Layer
    hp_filters = hp.Int('filters', min_value=16, max_value=64, step=16)
    hp_kernel_size = hp.Choice('kernel_size', values=[2, 3, 4])
    hp_pool_size = hp.Choice('pool_size', values=[2, 3])
    x = Conv1D(filters=hp_filters, kernel_size=hp_kernel_size, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=hp_pool_size)(x)
    
    # LSTM Layer
    hp_units = hp.Int('units', min_value=4, max_value=32, step=4)
    x = LSTM(units=hp_units, activation='relu', return_sequences=False)(x)
    
    # Dropout Layer
    hp_dropout = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    x = Dropout(hp_dropout)(x)
    
    # Output Layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]), 
                                 epsilon=1e-08, decay=0.0),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    return model

def setup_callbacks(log_dir):
    """Sets up TensorBoard and EarlyStopping callbacks."""
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
    return [tensorboard_callback, early_stopping]

def save_model(model, file_name):
    """Saves the Keras model."""
    model.save(f"{file_name}.keras")

def load_saved_model(file_name):
    """Loads a saved Keras model."""
    return load_model(f"{file_name}.keras")

def calculate_tpr_tnr_fpr_fnr(y_true, y_pred):
    """Calculates TPR, TNR, FPR, and FNR based on the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn)  # True Positive Rate (Sensitivity, Recall)
    tnr = tn / (tn + fp)  # True Negative Rate (Specificity)
    fpr = fp / (fp + tn)  # False Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate

    return tpr, tnr, fpr, fnr


