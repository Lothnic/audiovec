"""Training pipeline for the audiovec embedding model."""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from audiovec.config import EMBEDDING_DIM, EPOCHS, BATCH_SIZE, LEARNING_RATE
from audiovec.model import build_model


def prepare_labels(y: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    """Encode integer emotion labels (1–8) to zero-indexed categorical vectors."""
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)           # 0 … num_classes-1
    y_categorical = to_categorical(y_encoded)
    return y_categorical, encoder


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    input_shape: tuple[int, int, int],
    embedding_dim: int = EMBEDDING_DIM,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    verbose: int = 1,
) -> tuple[tf.keras.Model, LabelEncoder, np.ndarray]:
    """Full training loop: prepare labels, build model, fit, return artefacts.

    Returns
    -------
    trained_model : tf.keras.Model
    label_encoder : LabelEncoder
    y_categorical : np.ndarray
    """
    y_categorical, encoder = prepare_labels(y)

    num_classes = y_categorical.shape[1]
    model = build_model(
        input_shape=input_shape,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(X, y_categorical, epochs=epochs, batch_size=batch_size, verbose=verbose)

    return model, encoder, y_categorical
