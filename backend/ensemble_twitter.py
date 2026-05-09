"""Average softmax predictions of the three Twitter models on the same
clean resplit val set. This is a cheap ensemble that doesn't need retraining.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402

from config import HYPERPARAMS, seed_tensorflow  # noqa: E402
from src import data_loader, evaluate, preprocessing  # noqa: E402


def main():
    seed_tensorflow()
    from tensorflow.keras.models import load_model

    hp = HYPERPARAMS["sentiment"].copy()
    hp.update({"resplit": True, "embedding_mask_zero": True})

    raw = data_loader.load("sentiment", "twitter")
    data = preprocessing.preprocess("sentiment", "twitter", raw, hp)

    # mask_zero serialization is Keras-3 hostile — retrain each architecture
    # in-memory rather than load from .h5 (cheap: ~1 min/model on the cleaned
    # data with the sweet-spot HP from twitter_hp2).
    from src import models as model_builders
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    probas = []
    for mdl in ("RNN", "LSTM", "GRU"):
        seed_tensorflow()
        builder = model_builders.BUILDERS[mdl]
        keras_model = builder(
            input_shape=data["input_shape"],
            output_dim=data["output_dim"],
            hp=hp,
            classify=data["classify"],
        )
        keras_model.fit(
            data["X_train"], data["y_train"],
            validation_data=(data["X_val"], data["y_val"]),
            epochs=15, batch_size=hp["batch_size"],
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0),
            ],
            verbose=0,
        )
        p = keras_model.predict(data["X_val"], verbose=0)
        probas.append(p)
        print(f"  trained {mdl} — pred shape {p.shape}")

    avg = np.mean(probas, axis=0)
    y_pred = np.argmax(avg, axis=1)
    metrics = evaluate.evaluate_classification(
        data["y_val"], y_pred, avg, data["classes"]
    )
    print()
    print("Twitter ensemble (avg of RNN+LSTM+GRU softmax):")
    print(f"  acc = {metrics['accuracy']}")
    print(f"  f1  = {metrics['f1']}")
    print(f"  prec= {metrics['precision']}")
    print(f"  rec = {metrics['recall']}")
    print(f"  auc = {metrics['auc']}")

    # Compare individual models
    print("\nFor reference (individual):")
    for mdl, p in zip(("RNN", "LSTM", "GRU"), probas):
        y_p = np.argmax(p, axis=1)
        m_ind = evaluate.evaluate_classification(data["y_val"], y_p, p, data["classes"])
        print(f"  {mdl}: acc={m_ind['accuracy']} f1={m_ind['f1']:.4f}")


if __name__ == "__main__":
    main()
