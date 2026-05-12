import tensorflow as tf
import json
import tempfile
from calamari_ocr.ocr.training.params import TrainerParams
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr.training.trainer import Trainer

checkpoint_path = r"E:\Git\samuTrain.py\models\generic_ocr_model\checkpoint\checkpoint_0032\trainer_params.json"

with open(checkpoint_path) as f:
    params_dict = json.load(f)
    params = TrainerParams.from_dict(params_dict)

temp_dir = tempfile.mkdtemp()
params.output_dir = temp_dir

scenario = CalamariScenario(params.scenario)
trainer = Trainer(params, scenario, restore=True)
graph = scenario.graph

print(f"Model loaded successfully. Graph: {graph}")
