import os
import numpy as np
import tensorflow as tf
import argparse
import glob
from typing import Dict, Any, List, Tuple
from PIL import Image

# Calamari belső komponensek
from calamari_ocr.ocr.model.model import Model
from calamari_ocr.ocr.training.trainer import Trainer
from calamari_ocr.ocr.predict.predictor import Predictor
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr.training.params import TrainerParams
from calamari_ocr.ocr.training.pipeline_params import CalamariDefaultTrainerPipelineParams

class CalamariManualEngine:
    def __init__(self, sCheckpointPath: str):
        """
        Inicializálja a motort. A modellt egyszer tölti be a memóriába.
        """
        assert os.path.exists(sCheckpointPath), f"Checkpoint nem található: {sCheckpointPath}"
        
        # 1. Trainer betöltése a konfiguráció és a processzorok miatt
        # Nem indítjuk el a .train()-t, csak kiszedjük belőle az agyat.
        trainer_params = CalamariScenario.default_trainer_params()
        trainer_params.warmstart.model = sCheckpointPath
        trainer_params.warmstart.allow_partial = True
        
        # Create trainer and extract components using scenario class method
        self.oTrainer: Trainer = CalamariScenario.create_trainer(trainer_params)
        self.oModel: Model = self.oTrainer._model
        
        # Setup data to get codec and processor
        self.oTrainer.setup_data()
        data = self.oTrainer._data
        self.oCodec = data.params.codec
        
        # Create data processor with correct pipeline parameters
        train_pipeline_params = CalamariDefaultTrainerPipelineParams()
        train_pipeline_params.setup.mode = "training"
        self.oDataProcessor = data.params.pre_proc.create(train_pipeline_params.setup, data.params)
        
        # 2. Megállító flag
        self.bStopRequested: bool = False

    def train_single_case(self, rImage: np.ndarray, sGroundTruth: str) -> Dict[str, Any]:
        """
        Egyetlen kép alapú súlyfrissítés a memóriában.
        Visszatér: {'loss': float, 'pred': str, 'cer': float}
        """
        assert rImage is not None, "A bemeneti kép (rImage) hiányzik."
        assert len(sGroundTruth) > 0, "A Ground Truth (sGroundTruth) nem lehet üres."
        
        if self.bStopRequested:
            return {"status": "stopped", "loss": 0.0, "pred": "", "cer": 1.0}

        # --- STEP 1: Pre-processing (Memóriában) ---
        # A Calamari belső logikájával normalizáljuk a képet (magasság, binárisítás stb.)
        oSample = self.oDataProcessor.process_single(rImage, sGroundTruth)
        
        # TensorFlow elvárja a batch dimenziót (1, Magasság, Szélesség, Csatorna)
        rInputX = np.expand_dims(oSample.x, axis=0)
        rInputY = np.expand_dims(oSample.y, axis=0)

        # --- STEP 2: Tanulás (Weight Update) ---
        # Ez a lényeg: közvetlen súlyfrissítés a GPU/CPU memóriában
        rMetrics = self.oModel.train_on_batch(rInputX, rInputY)
        fLoss = float(rMetrics[0]) if isinstance(rMetrics, (list, np.ndarray)) else float(rMetrics)

        # --- STEP 3: Predikció (UI visszajelzéshez) ---
        # Frissítés után azonnal megnézzük, mit "gondol" most a modell
        rPredictionRaw = self.oModel.predict(rInputX)
        sPred = self.oCodec.decode(rPredictionRaw[0])
        
        fCer = self._calculate_cer(sPred, sGroundTruth)

        return {
            "status": "ok",
            "loss": fLoss,
            "pred": sPred,
            "cer": fCer
        }

    def _calculate_cer(self, sPred: str, sTarget: str) -> float:
        """Karakter hibaarány (CER) számítása edit distance alapján."""
        import editdistance
        iDistance = editdistance.eval(sPred, sTarget)
        return float(iDistance / max(len(sTarget), 1))

    def save_current_state(self, sOutputPath: str):
        """Elmenti a memóriában lévő frissített súlyokat a lemezre."""
        assert sOutputPath is not None
        # A Calamari saját mentési logikáját hívjuk meg
        self.oModel.save_weights(sOutputPath)

    def request_stop(self):
        self.bStopRequested = True


def load_dataset(data_dir: str) -> List[Tuple[np.ndarray, str]]:
    """Load all image-ground truth pairs from a directory"""
    dataset = []
    
    # Find all .bin.png files
    image_files = glob.glob(os.path.join(data_dir, "*.bin.png"))
    
    for image_path in image_files:
        # Get corresponding .gt.txt file
        gt_path = image_path.replace('.bin.png', '.gt.txt')
        
        if os.path.exists(gt_path):
            try:
                # Load image
                image = Image.open(image_path).convert('L')
                image_np = np.array(image)
                
                # Load ground truth
                with open(gt_path, 'r', encoding='utf-8') as f:
                    ground_truth = f.read().strip()
                
                if len(ground_truth) > 0:  # Skip empty ground truths
                    dataset.append((image_np, ground_truth))
                    
            except Exception as e:
                print(f"⚠️  Error loading {image_path}: {e}")
                
    print(f"📂 Loaded {len(dataset)} training samples from {data_dir}")
    return dataset


def train_multiple_epochs(engine: CalamariManualEngine, dataset: List[Tuple[np.ndarray, str]], 
                        epochs: int = 10, output_model: str = None) -> Dict[str, Any]:
    """Train for multiple epochs with data persistence between epochs"""
    
    print(f"\n🚀 Starting {epochs}-epoch training with {len(dataset)} samples")
    print("=" * 60)
    
    epoch_results = []
    
    for epoch in range(epochs):
        print(f"\n📊 Epoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        epoch_loss = 0.0
        epoch_cer = 0.0
        successful_training = 0
        
        # Shuffle dataset for each epoch
        import random
        shuffled_dataset = dataset.copy()
        random.shuffle(shuffled_dataset)
        
        for i, (image, ground_truth) in enumerate(shuffled_dataset):
            print(f"  🖼️  Training on sample {i+1}/{len(shuffled_dataset)}: '{ground_truth}'")
            
            # Train on this sample
            result = engine.train_single_case(image, ground_truth)
            
            if result['status'] == 'ok':
                epoch_loss += result['loss']
                epoch_cer += result['cer']
                successful_training += 1
                print(f"     ✅ Loss: {result['loss']:.4f}, CER: {result['cer']:.4f}, Pred: '{result['pred']}'")
            else:
                print(f"     ❌ Training failed: {result['status']}")
        
        # Calculate epoch averages
        if successful_training > 0:
            avg_loss = epoch_loss / successful_training
            avg_cer = epoch_cer / successful_training
        else:
            avg_loss = 0.0
            avg_cer = 1.0
            
        epoch_results.append({
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'avg_cer': avg_cer,
            'successful': successful_training,
            'total': len(shuffled_dataset)
        })
        
        print(f"\n📈 Epoch {epoch + 1} Summary:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Average CER: {avg_cer:.4f}")
        print(f"   Success Rate: {successful_training}/{len(shuffled_dataset)} ({100*successful_training/len(shuffled_dataset):.1f}%)")
    
    # Save model if requested
    if output_model:
        print(f"\n💾 Saving model to: {output_model}")
        os.makedirs(output_model, exist_ok=True)
        engine.save_current_state(os.path.join(output_model, "manual_trained_weights.ckpt"))
        print("✅ Model saved successfully!")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Training completed!")
    print("=" * 60)
    
    for result in epoch_results:
        print(f"Epoch {result['epoch']:2d}: Loss={result['avg_loss']:.4f}, CER={result['avg_cer']:.4f}, Success={result['successful']}/{result['total']}")
    
    return {
        'epoch_results': epoch_results,
        'total_epochs': epochs,
        'total_samples': len(dataset)
    }


def main():
    """Main function with default arguments"""
    parser = argparse.ArgumentParser(description='Calamari Manual Engine - Multi-Epoch Training')
    
    # Default arguments for meaningful operation
    parser.add_argument('--checkpoint', type=str, 
                       default='models/generic_ocr_model/best.ckpt.json',
                       help='Path to checkpoint file (default: models/generic_ocr_model/best.ckpt.json)')
    
    parser.add_argument('--data-dir', type=str,
                       default='data/64_case',
                       help='Directory containing training data (default: data/64_case)')
    
    parser.add_argument('--epochs', type=int,
                       default=10,
                       help='Number of training epochs (default: 10)')
    
    parser.add_argument('--output-model', type=str,
                       default='models/manual_engine_10_epoch',
                       help='Output directory for trained model (default: models/manual_engine_10_epoch)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return 1
    
    try:
        # Initialize manual engine
        print(f"🔧 Initializing manual engine with checkpoint: {args.checkpoint}")
        engine = CalamariManualEngine(args.checkpoint)
        print("✅ Manual engine initialized successfully!")
        
        # Load dataset
        dataset = load_dataset(args.data_dir)
        
        if len(dataset) == 0:
            print(f"❌ No valid training data found in {args.data_dir}")
            return 1
        
        # Train for multiple epochs
        results = train_multiple_epochs(
            engine=engine,
            dataset=dataset,
            epochs=args.epochs,
            output_model=args.output_model
        )
        
        print(f"\n🏁 Training completed successfully!")
        print(f"   - Total epochs: {results['total_epochs']}")
        print(f"   - Total samples: {results['total_samples']}")
        print(f"   - Model saved to: {args.output_model}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())