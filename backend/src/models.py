"""Keras model builders — one per architecture family.

Each builder takes ``input_shape``, ``output_dim`` (1 for regression, n_classes
for classification), ``hp`` (the per-task hyperparameter dict from config.py),
and ``classify`` (bool). They return a compiled ``tf.keras.Model`` ready to
``.fit``.

For text tasks, ``hp`` includes ``vocab_size`` and ``embedding_dim`` and the
input is a 1-D sequence of token IDs that goes through an Embedding layer.
"""

from __future__ import annotations

from typing import Any

from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.optimizers import Adam


def _resolve_units(spec, fallback: int) -> list[int]:
    if spec is None:
        return [fallback]
    if isinstance(spec, list):
        return spec
    return [int(spec)]


def _input_layer(input_shape, hp: dict[str, Any]):
    """Add an Embedding layer for text tasks; otherwise plain Input."""
    if "vocab_size" in hp and len(input_shape) == 1:
        inp = layers.Input(shape=input_shape, dtype="int32")
        x = layers.Embedding(
            input_dim=hp["vocab_size"],
            output_dim=hp["embedding_dim"],
            mask_zero=True,
        )(inp)
        return inp, x
    inp = layers.Input(shape=input_shape)
    return inp, inp


def _head(x, output_dim: int, classify: bool):
    if classify:
        return layers.Dense(output_dim, activation="softmax")(x)
    return layers.Dense(1, activation="linear")(x)


def _compile(model: Model, classify: bool, lr: float) -> Model:
    if classify:
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    else:
        loss = "mse"
        metrics = ["mae"]
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss, metrics=metrics)
    return model


# ── ANN ─────────────────────────────────────────────────────────────────────
def build_ann(input_shape, output_dim: int, hp: dict[str, Any], classify: bool) -> Model:
    """Dense MLP. Flattens any sequence input then runs through dense stack."""
    inp, x = _input_layer(input_shape, hp)
    if len(x.shape) > 2:
        x = layers.Flatten()(x)

    units = _resolve_units(hp.get("hidden_units", {}).get("ANN"), 128)
    if not isinstance(units, list):
        units = [units]
    if len(units) == 1:
        units = [units[0], units[0] // 2]

    for u in units:
        x = layers.Dense(u, activation="relu")(x)
        x = layers.Dropout(hp["dropout"])(x)

    out = _head(x, output_dim, classify)
    return _compile(Model(inp, out, name="ANN"), classify, hp["learning_rate"])


def _maybe_pool_over_time(x, hp: dict[str, Any], force: bool = False):
    """If ``hp.pool_over_time`` (or ``force``), keep ``return_sequences=True`` on
    the last recurrent layer and pool across time. Otherwise return as-is."""
    if hp.get("pool_over_time") or force:
        return layers.GlobalAveragePooling1D()(x)
    return x


# ── Vanilla RNN ─────────────────────────────────────────────────────────────
def build_rnn(input_shape, output_dim: int, hp: dict[str, Any], classify: bool) -> Model:
    """SimpleRNN. Used for all three tasks."""
    inp, x = _input_layer(input_shape, hp)
    units = _resolve_units(hp.get("hidden_units", {}).get("RNN"), 64)
    pool = bool(hp.get("pool_over_time"))
    for i, u in enumerate(units):
        return_seq = pool or i < len(units) - 1
        x = layers.SimpleRNN(u, activation="tanh", return_sequences=return_seq,
                             dropout=hp["dropout"])(x)
    if pool:
        x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(hp["dropout"])(x)
    out = _head(x, output_dim, classify)
    return _compile(Model(inp, out, name="RNN"), classify, hp["learning_rate"])


# ── LSTM ───────────────────────────────────────────────────────────────────
def build_lstm(input_shape, output_dim: int, hp: dict[str, Any], classify: bool) -> Model:
    """Stacked LSTM. The default hp.hidden_units.LSTM is a list (e.g. [128, 64])."""
    inp, x = _input_layer(input_shape, hp)
    units = _resolve_units(hp.get("hidden_units", {}).get("LSTM"), 128)
    pool = bool(hp.get("pool_over_time"))
    for i, u in enumerate(units):
        return_seq = pool or i < len(units) - 1
        x = layers.LSTM(
            u, return_sequences=return_seq,
            dropout=hp["dropout"], recurrent_dropout=0.0,
            kernel_regularizer=regularizers.l2(1e-4),
        )(x)
    if pool:
        x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(hp["dropout"])(x)
    if classify and len(units) > 1:
        x = layers.Dense(32, activation="relu")(x)
    out = _head(x, output_dim, classify)
    return _compile(Model(inp, out, name="LSTM"), classify, hp["learning_rate"])


# ── GRU ────────────────────────────────────────────────────────────────────
def build_gru(input_shape, output_dim: int, hp: dict[str, Any], classify: bool) -> Model:
    """Stacked GRU. For long sequences (TESS: 174 frames), use
    ``hp.pool_over_time=True`` so the final hidden state isn't the only signal —
    without that, GRU+adam+lr=1e-3 reliably gets stuck at chance."""
    inp, x = _input_layer(input_shape, hp)
    units = _resolve_units(hp.get("hidden_units", {}).get("GRU"), 96)
    pool = bool(hp.get("pool_over_time"))
    for i, u in enumerate(units):
        return_seq = pool or i < len(units) - 1
        layer_dropout = hp["dropout"] if i == 0 else 0.0
        x = layers.GRU(u, return_sequences=return_seq, dropout=layer_dropout)(x)
    if pool:
        x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(hp["dropout"])(x)
    if classify and len(units) > 1:
        x = layers.Dense(32, activation="relu")(x)
    out = _head(x, output_dim, classify)
    return _compile(Model(inp, out, name="GRU"), classify, hp["learning_rate"])


BUILDERS = {
    "ANN": build_ann,
    "RNN": build_rnn,
    "LSTM": build_lstm,
    "GRU": build_gru,
}
