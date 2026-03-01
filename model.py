"""
model.py — Definición de la arquitectura, pérdidas y métricas del modelo DocVerify.
"""

import math

import tensorflow as tf
from tensorflow.keras import layers, models

import config


# ============================================================
# PÉRDIDAS
# ============================================================

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Dice Loss por imagen. Devuelve un tensor de shape (B,).
    Diseñada para manejar el desbalanceo de clases a nivel de pixel.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    inter = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)

    return 1.0 - (2.0 * inter + smooth) / (denom + smooth)


_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Pérdida combinada para segmentación:
      BCE(y_true, y_pred) + mean(DiceLoss(y_true, y_pred))
    """
    return _bce(y_true, y_pred) + tf.reduce_mean(dice_loss(y_true, y_pred))


# ============================================================
# MÉTRICAS
# ============================================================

class DiceGlobal(tf.keras.metrics.Metric):
    """
    Dice acumulado sobre todo el dataset (global), no por batch.
    Compatible con el threshold configurable THR_MASK.
    """

    def __init__(self, name: str = "dice_global", threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.inter = self.add_weight(name="inter", initializer="zeros", dtype=tf.float32)
        self.denom = self.add_weight(name="denom", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true > 0.5,          tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        self.inter.assign_add(tf.reduce_sum(y_true * y_pred))
        self.denom.assign_add(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

    def result(self) -> tf.Tensor:
        return (2.0 * self.inter + 1e-6) / (self.denom + 1e-6)

    def reset_states(self):
        self.inter.assign(0.0)
        self.denom.assign(0.0)


# ============================================================
# CALLBACKS
# ============================================================

class EpochCounter(tf.keras.callbacks.Callback):
    """Registra cuántos epochs completó el entrenamiento (útil con EarlyStopping)."""

    def __init__(self):
        super().__init__()
        self.epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs = int(epoch) + 1


class AddValDistance(tf.keras.callbacks.Callback):
    """
    Calcula en cada epoch la distancia Euclídea al punto ideal (1, 1)
    en el espacio PR-AUC × Dice y la añade a los logs como
    'val_distance_to_ideal'. Permite usar EarlyStopping sobre esta métrica.
    """

    def __init__(
        self,
        pr_key:   str = "val_cls_prauc",
        dice_key: str = "val_mask_dice_global",
    ):
        super().__init__()
        self.pr_key   = pr_key
        self.dice_key = dice_key

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        pr = logs.get(self.pr_key)
        dc = logs.get(self.dice_key)
        if pr is None or dc is None:
            return
        logs["val_distance_to_ideal"] = float(
            math.sqrt((1.0 - float(pr)) ** 2 + (1.0 - float(dc)) ** 2)
        )


# ============================================================
# ARQUITECTURA
# ============================================================

def build_model(
    input_shape: tuple = (224, 224, 3),
    alpha:        float = 0.2,
    dropout_rate: float = 0.5,
    dec_ch:       int   = 128,
) -> tf.keras.Model:
    """
    Patel CNN Encoder + U-Net Decoder para clasificación y segmentación multi-tarea.

    Entradas:
      input_shape   — Dimensiones de la imagen de entrada (H, W, C)
      alpha         — Pendiente negativa de LeakyReLU
      dropout_rate  — Tasa de dropout en la cabeza de clasificación
      dec_ch        — Canales base del decoder (se reduce por 2 en cada nivel)

    Salidas (dict):
      "cls"  — (B, 1)       sigmoid: probabilidad de ataque
      "mask" — (B, H, W, 1) sigmoid: máscara de regiones alteradas
    """
    inp = layers.Input(shape=input_shape, name="input_patch")

    # ── Encoder ──────────────────────────────────────────────

    # Bloque 1: 8 filtros, sin pooling
    x = layers.Conv2D(8, (3, 3), padding="same", name="enc_conv1_8")(inp)
    x = layers.LeakyReLU(negative_slope=alpha)(x)
    x = layers.BatchNormalization(name="enc_bn1")(x)

    # Bloque 2: 16 filtros → skip s224
    x = layers.Conv2D(16, (3, 3), padding="same", name="enc_conv2_16a")(x)
    x = layers.LeakyReLU(negative_slope=alpha)(x)
    x = layers.Conv2D(16, (3, 3), padding="same", name="enc_conv2_16b")(x)
    x = layers.LeakyReLU(negative_slope=alpha)(x)
    s224 = layers.BatchNormalization(name="enc_bn2")(x)
    x = layers.AveragePooling2D((2, 2), name="enc_pool1")(s224)   # → 112×112

    # Bloque 3: 32 filtros × 3 → skip s112
    for i in range(3):
        x = layers.Conv2D(32, (3, 3), padding="same", name=f"enc_conv3_32_{i}")(x)
        x = layers.LeakyReLU(negative_slope=alpha)(x)
    s112 = layers.BatchNormalization(name="enc_bn3")(x)
    x = layers.AveragePooling2D((2, 2), name="enc_pool2")(s112)   # → 56×56

    # Bloque 4: 64 filtros × 4 → skip s56
    for i in range(4):
        x = layers.Conv2D(64, (3, 3), padding="same", name=f"enc_conv4_64_{i}")(x)
        x = layers.LeakyReLU(negative_slope=alpha)(x)
    s56 = layers.BatchNormalization(name="enc_bn4")(x)
    x = layers.AveragePooling2D((2, 2), name="enc_pool3")(s56)    # → 28×28

    # Bloque 5: 128 filtros → skip s28
    x = layers.Conv2D(128, (5, 5), padding="same", name="enc_conv5_128")(x)
    x = layers.LeakyReLU(negative_slope=alpha)(x)
    s28 = layers.BatchNormalization(name="enc_bn5")(x)
    x = layers.MaxPooling2D((2, 2), name="enc_pool4")(s28)         # → 14×14

    # Bloque 6: 256 filtros → skip s14 → bottleneck 7×7
    x = layers.Conv2D(256, (5, 5), padding="same", name="enc_conv6_256")(x)
    x = layers.LeakyReLU(negative_slope=alpha)(x)
    s14 = layers.BatchNormalization(name="enc_bn6")(x)
    bottleneck = layers.MaxPooling2D((2, 2), name="enc_pool5")(s14)  # → 7×7

    # ── Cabeza de clasificación ───────────────────────────────
    c = layers.GlobalAveragePooling2D(name="cls_gap")(bottleneck)
    c = layers.Dropout(dropout_rate, name="cls_do1")(c)
    c = layers.Dense(32,  name="cls_fc1")(c); c = layers.LeakyReLU(negative_slope=alpha)(c)
    c = layers.Dropout(dropout_rate, name="cls_do2")(c)
    c = layers.Dense(16,  name="cls_fc2")(c); c = layers.LeakyReLU(negative_slope=alpha)(c)
    c = layers.Dropout(dropout_rate, name="cls_do3")(c)
    c = layers.Dense(16,  name="cls_fc3")(c); c = layers.LeakyReLU(negative_slope=alpha)(c)
    c = layers.Dropout(dropout_rate, name="cls_do4")(c)
    cls_out = layers.Activation("sigmoid", name="cls")(layers.Dense(1, name="cls_logits")(c))

    # ── Cabeza de segmentación (decoder U-Net) ────────────────
    def dec_block(x, skip, ch, name):
        x = layers.UpSampling2D((2, 2), interpolation="bilinear", name=f"{name}_up")(x)
        x = layers.Concatenate(name=f"{name}_cat")([x, skip])
        x = layers.Conv2D(ch, 3, padding="same", name=f"{name}_conv1")(x)
        x = layers.LeakyReLU(negative_slope=alpha)(x)
        x = layers.Conv2D(ch, 3, padding="same", name=f"{name}_conv2")(x)
        x = layers.LeakyReLU(negative_slope=alpha)(x)
        return x

    m = layers.LeakyReLU(negative_slope=alpha)(
        layers.Conv2D(dec_ch, 1, padding="same", name="mask_proj")(bottleneck)
    )
    m = dec_block(m, s14,  dec_ch,       "dec14")   # 7  → 14
    m = dec_block(m, s28,  dec_ch // 2,  "dec28")   # 14 → 28
    m = dec_block(m, s56,  dec_ch // 4,  "dec56")   # 28 → 56
    m = dec_block(m, s112, dec_ch // 8,  "dec112")  # 56 → 112
    m = dec_block(m, s224, dec_ch // 16, "dec224")  # 112 → 224

    m = layers.Resizing(input_shape[0], input_shape[1],
                        interpolation="bilinear", name="mask_resize")(m)
    mask_out = layers.Activation("sigmoid", name="mask")(
        layers.Conv2D(1, 1, padding="same", name="mask_logits")(m)
    )

    return models.Model(inp, outputs={"cls": cls_out, "mask": mask_out},
                        name="docverify_multitask")


def build_and_compile(params: dict) -> tf.keras.Model:
    """Construye y compila el modelo con los hiperparámetros dados."""
    model = build_model(
        input_shape  = (config.PATCH_SIZE, config.PATCH_SIZE, 3),
        alpha        = float(params["alpha"]),
        dropout_rate = float(params["dropout_rate"]),
        dec_ch       = int(params["dec_ch"]),
    )

    try:
        opt = tf.keras.optimizers.AdamW(
            learning_rate=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=float(params["lr"]))

    prauc = tf.keras.metrics.AUC(curve="PR", name="prauc", num_thresholds=200)

    model.compile(
        optimizer    = opt,
        loss         = {"cls": tf.keras.losses.BinaryCrossentropy(), "mask": bce_dice_loss},
        loss_weights = {"cls": 1.0, "mask": float(params["loss_w_mask"])},
        metrics      = {
            "cls":  [prauc],
            "mask": [DiceGlobal(name="dice_global", threshold=config.THR_MASK)],
        },
    )
    return model


def build_and_compile_variant(params: dict, variant: str) -> tf.keras.Model:
    """
    Igual que build_and_compile pero con pesos de pérdida según la variante de ablación.

    Variantes:
      "multitask"         — cls=1.0, mask=loss_w_mask  (configuración completa)
      "cls_only"          — cls=1.0, mask=0.0
      "seg_only"          — cls=0.0, mask=loss_w_mask
      "unweighted_losses" — cls=1.0, mask=1.0
    """
    lw_mask = float(params.get("loss_w_mask", 1.0))

    loss_weights = {
        "multitask":         {"cls": 1.0, "mask": lw_mask},
        "cls_only":          {"cls": 1.0, "mask": 0.0},
        "seg_only":          {"cls": 0.0, "mask": lw_mask},
        "unweighted_losses": {"cls": 1.0, "mask": 1.0},
    }

    if variant not in loss_weights:
        raise ValueError(f"Variante desconocida: '{variant}'. "
                         f"Opciones: {list(loss_weights)}")

    model = build_model(
        input_shape  = (config.PATCH_SIZE, config.PATCH_SIZE, 3),
        alpha        = float(params["alpha"]),
        dropout_rate = float(params["dropout_rate"]),
        dec_ch       = int(params["dec_ch"]),
    )

    try:
        opt = tf.keras.optimizers.AdamW(
            learning_rate=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=float(params["lr"]))

    prauc = tf.keras.metrics.AUC(curve="PR", name="prauc", num_thresholds=200)

    model.compile(
        optimizer    = opt,
        loss         = {"cls": tf.keras.losses.BinaryCrossentropy(), "mask": bce_dice_loss},
        loss_weights = loss_weights[variant],
        metrics      = {
            "cls":  [prauc],
            "mask": [DiceGlobal(name="dice_global", threshold=config.THR_MASK)],
        },
    )
    return model
