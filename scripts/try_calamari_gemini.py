import os
import logging
import tempfile
import numpy as np
import tensorflow as tf
import cv2
import json

# TensorFlow/Keras figyelmeztetések elnyomása
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Calamari belső importok
from calamari_ocr.ocr.training.params import TrainerParams
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr.training.trainer import Trainer
from calamari_ocr.ocr.model.ctcdecoder.ctc_decoder import create_ctc_decoder

# ========== CONSTANTS (Bekötve a kért útvonalak) ==========
MODEL_DIR = r"E:\Git\samuTrain.py\models\new_model\best.ckpt"  # Alapnév
DATA_DIR = r"E:\Git\samuTrain.py\data\single_case"
SAMPLE_IMAGE_NAME = "01002c.bin.png"
OUTPUT_CHECKPOINT_DIR = r"E:\Git\samuTrain.py\models\updated_model"
LEARNING_RATE = 0.001
IMAGE_HEIGHT = 64 

class CalamariCharacterLearner:
    def __init__(self, model_base_path):
        from tfaip.trainer.scheduler import Constant  # Ez az import szükséges a javításhoz
        
        # JSON útvonal kezelése
        if model_base_path.endswith(".json"):
            json_path = model_base_path
        else:
            json_path = model_base_path + ".json"

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"A keresett JSON nem található: {json_path}")
        
        print(f"DEBUG: JSON betöltése innen: {json_path}")

        # 1. Paraméterek betöltése
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = f.read()
            self.params = TrainerParams.from_json(json_data)
        
        self.params.output_dir = tempfile.mkdtemp(prefix="calamari_ft_")
        
        # --- JAVÍTÁS ---
        # Az optimizer.learning_rate nem létezik. 
        # Helyette a params.learning_rate-et kell beállítani egy Constant schedulerre.
        self.params.learning_rate = Constant(value=LEARNING_RATE)
        # ---------------
        
        # 2. Modell és Trainer felépítése
        self.scenario = CalamariScenario(self.params.scenario)
        self.trainer = Trainer(self.params, self.scenario, restore=True)
        self.model = self.trainer.model
        
        # 3. Dekóder és segédváltozók
        self.codec = self.trainer.scenario.data.params.codec
        self.decoder = create_ctc_decoder(
            self.codec, 
            self.params.scenario.model.ctc_decoder_params
        )
        
        self.ds_factor = self.trainer.scenario.data.params.downscale_factor

    def cut_and_label(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return []

        # Fix magasságra skálázás a modellhez
        if img.shape[0] != IMAGE_HEIGHT:
            img = cv2.resize(img, (int(img.shape[1] * IMAGE_HEIGHT / img.shape[0]), IMAGE_HEIGHT))

        img_norm = img.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_norm, axis=(0, -1))

        # Predikció
        outputs = self.model.predict({"img": img_input})
        softmax = outputs["blank_last_softmax"][0]
        
        # Calamari vágási logika (CTC alapú)
        prediction = self.decoder.decode(softmax)
        
        fragments = []
        for pos in prediction.positions:
            start_px = pos.local_start * self.ds_factor
            end_px = (pos.local_end + 1) * self.ds_factor
            
            char_patch = img[:, start_px:end_px]
            
            if pos.chars and char_patch.size > 0:
                char_label = self.codec.code2char[pos.chars[0].label]
                fragments.append((char_patch, char_label))
                
        return fragments

    def train_on_fragment(self, char_img, char_label):
        # A patch-et is 64px magasra hozzuk
        char_img_res = cv2.resize(char_img, (max(1, char_img.shape[1]), IMAGE_HEIGHT))
        img_input = np.expand_dims(char_img_res, axis=(0, -1)).astype(np.float32) / 255.0
        
        try:
            label_id = self.codec.char2code[char_label]
        except KeyError:
            return None

        target = np.array([[label_id]], dtype=np.int32)

        with tf.GradientTape() as tape:
            outputs = self.model({"img": img_input}, training=True)
            logits = outputs["blank_last_logits"]
            reduced_logits = tf.reduce_mean(logits, axis=1) # Időbeli átlagolás
            
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                target, reduced_logits, from_logits=True
            )

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.trainer.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss.numpy()[0]

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        # best.ckpt mentése (ez létrehozza a .json-t is mellé)
        self.trainer.save_checkpoint(os.path.join(path, "best.ckpt"))

if __name__ == "__main__":
    try:
        full_image_path = os.path.join(DATA_DIR, SAMPLE_IMAGE_NAME)
        
        learner = CalamariCharacterLearner(MODEL_DIR)
        
        print(f"Karakterek kinyerése: {full_image_path}")
        fragments = learner.cut_and_label(full_image_path)
        
        print(f"Talált karakterek száma: {len(fragments)}")
        for i, (patch, label) in enumerate(fragments):
            l = learner.train_on_fragment(patch, label)
            if l is not None:
                print(f"Tanítás: '{label}' | Loss: {l:.4f}")
        
        learner.save(OUTPUT_CHECKPOINT_DIR)
        print(f"Sikeres mentés: {OUTPUT_CHECKPOINT_DIR}")
        
    except Exception as e:
        print(f"\nHIBA TÖRTÉNT: {e}")
        import traceback
        traceback.print_exc()