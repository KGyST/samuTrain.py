"""
Script to run a SINGLE training event on a Calamari OCR model,
update weights, and save a new checkpoint.
Uses:
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow C++ warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow Python warnings
logging.getLogger('keras').setLevel(logging.ERROR)  # Suppress Keras warnings

import numpy as np
import tensorflow as tf
from PIL import Image
from calamari_ocr.ocr.dataset.codec import Codec

# ========== CONSTANTS (EDIT ONLY THESE IF NEEDED) ==========
MODEL_DIR = r"E:\Git\samuTrain.py\models\new_model\best.ckpt"  # Path to best.ckpt (SavedModel)
DATA_DIR = r"E:\Git\samuTrain.py\data\single_case"                # Folder with .png + .gt.txt
SAMPLE_IMAGE_NAME = "01002c.bin.png"                           # Image filename
OUTPUT_CHECKPOINT_DIR = r"E:\Git\samuTrain.py\models\updated_model"  # New SavedModel dir
LEARNING_RATE = 0.001                                          # Optimizer learning rate
IMAGE_HEIGHT = 64                                             # Must match model's expected height
CHARSET_FILE = os.path.join(os.path.dirname(MODEL_DIR), "extended_charset.txt")  # Path to charset file

# ========== HELPER FUNCTIONS ==========
def preprocess_image(image_path, target_height=IMAGE_HEIGHT):
    """Load and preprocess a single image for Calamari OCR."""
    img = Image.open(image_path).convert("L")  # Grayscale
    img = img.resize((img.width, target_height), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Shape: [1, height, width, 1]
    return img

def read_label(label_path):
    """Read ground truth text from .gt.txt file."""
    with open(label_path, "r") as f:
        return f.read().strip()  # Remove newline

def read_charset(charset_path):
    """Read the character set from a file."""
    with open(charset_path, "r", encoding="utf-8") as f:
        return f.read()

def train_single_step(model, optimizer, image, label_array, label_length, codec):
    """Run a single training step and return PRED and CONF."""
    # Add training flag as the second input
    training_flag = tf.constant(True, dtype=tf.bool)  # Shape: ()
    with tf.GradientTape() as tape:
        # Forward pass: Pass BOTH image and training_flag
        logits = model([image, training_flag], training=True)  # Shape: [1, width, num_chars+1]
        # CTC loss
        input_length = np.array([tf.shape(logits)[1]], dtype=np.int32)
        loss = tf.nn.ctc_loss(
            labels=label_array,
            logits=logits,
            input_length=input_length,
            label_length=label_length,
            blank_index=codec.blank  # Use 'blank' for Calamari OCR
        )
        loss = tf.reduce_mean(loss)

    # Backward pass: compute gradients and update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Decode prediction
    pred_text, conf = codec.decode(logits)
    return pred_text[0], conf[0]  # Return first (and only) prediction

def save_model(model, output_dir):
    """Save the model as a new SavedModel directory."""
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    print(f"✅ Model saved to {output_dir}")

# ========== MAIN PIPELINE ==========
def main():
    # --- Load Model ---
    print(f"🔍 Loading model from {MODEL_DIR}...")
    model = tf.keras.models.load_model(MODEL_DIR)

    # --- Load Codec ---
    print(f"🔍 Loading charset from {CHARSET_FILE}...")
    charset = read_charset(CHARSET_FILE)
    codec = Codec(charset)

    # --- Initialize Optimizer ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # --- Load Sample Data ---
    image_path = os.path.join(DATA_DIR, SAMPLE_IMAGE_NAME)
    label_path = os.path.join(DATA_DIR, SAMPLE_IMAGE_NAME.replace(".bin.png", ".gt.txt"))
    print(f"📄 Using image: {image_path}")
    print(f"📄 Using label: {label_path}")

    image = preprocess_image(image_path)
    label_text = read_label(label_path)
    print(f"🏷️  Label: '{label_text}'")

    # Encode label for CTC loss
    encoded = codec.encode(label_text)
    label_array = np.array([encoded], dtype=np.int32)  # Shape: [1, max_length]
    label_length = np.array([len(encoded)], dtype=np.int32)

    # --- Run Single Training Step ---
    print("🚀 Running single training step...")
    PRED, CONF = train_single_step(model, optimizer, image, label_array, label_length, codec)
    print(f"🎯 Prediction: '{PRED}', Confidence: {CONF:.4f}")

    # --- Save Updated Model ---
    print("💾 Saving updated model...")
    save_model(model, OUTPUT_CHECKPOINT_DIR)

if __name__ == "__main__":
    main()