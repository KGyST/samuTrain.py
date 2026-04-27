#!/usr/bin/env python3
# What is this file for: Manual training script with immediate weight updates and PRED/CERT feedback
# When it is created: 2026-04-27
# Purpose: Train on single case/batch using train_on_batch() for granular control

import os
import sys
import time
import argparse
import tempfile
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

# Add src/ and project root to sys.path
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectRoot = os.path.dirname(scriptDir)
srcDir = os.path.join(projectRoot, 'src')
sys.path.insert(0, srcDir)
sys.path.insert(0, projectRoot)

# Force minimal logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

class CalamariManualTrainer:
    def __init__(self, checkpoint_path: str):
        """Initialize manual trainer with existing checkpoint"""
        print(f"🔧 Loading checkpoint: {checkpoint_path}")
        
        try:
            # Try to use the optimized manual engine first
            from try_calamari_manual_engine import CalamariManualEngine
            self.engine = CalamariManualEngine(checkpoint_path)
            self.use_manual_engine = True
            self.checkpoint_path = checkpoint_path
            self.stop_requested = False
            print(f"✅ Manual engine loaded successfully")
        except Exception as e:
            print(f"⚠️  Manual engine failed: {e}")
            print(f"� Falling back to temporary training method")
            
            # Fallback to original method
            from calamari_ocr.ocr.predict.predictor import MultiPredictor
            self.predictor = MultiPredictor.from_paths(checkpoints=[checkpoint_path])
            self.engine = None
            self.use_manual_engine = False
            self.checkpoint_path = checkpoint_path
            self.model_dir = os.path.dirname(checkpoint_path)
            self.stop_requested = False
            print(f"✅ Fallback method loaded successfully")

    def train_single_case(self, image_path: str, sGroundTruth: str) -> dict:
        """
        Functional training: 1 image, 1 GT.
        Uses manual engine when available, falls back to temporary training.
        Returns PRED, CER and Loss values for the UI.
        """
        assert os.path.exists(image_path), f"Image file not found: {image_path}"
        assert len(sGroundTruth) > 0, "Ground truth cannot be empty"
        
        if self.stop_requested:
            return {"status": "stopped"}

        # Try manual engine first (much faster!)
        if self.use_manual_engine and self.engine:
            try:
                from PIL import Image
                image = Image.open(image_path).convert('L')
                image_np = np.array(image)
                
                # Get prediction BEFORE training
                pred_before = self._predict_with_engine(image_np)
                
                # Train using manual engine (in-memory, no restart)
                start_time = time.time()
                result = self.engine.train_single_case(image_np, sGroundTruth)
                elapsed = time.time() - start_time
                
                if result['status'] == 'ok':
                    return {
                        "loss": result['loss'],
                        "pred": result['pred'],
                        "cer": result['cer'],
                        "pred_before": pred_before,
                        "training_time": elapsed,
                        "status": "ok"
                    }
                else:
                    print(f"⚠️  Manual engine failed: {result['status']}")
                    # Fall back to temporary training method
                    
            except Exception as e:
                print(f"⚠️  Manual engine error: {e}")
                print(f"🔄 Falling back to temporary training method")
        
        # Fallback to original temporary training method
        return self._train_with_temporary_method(image_path, sGroundTruth)
    
    def _train_with_temporary_method(self, image_path: str, sGroundTruth: str) -> dict:
        """Original temporary training method as fallback"""
        try:
            # Create temporary training data
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_data_dir = os.path.join(temp_dir, "temp_data")
                temp_model_dir = os.path.join(temp_dir, "temp_model")
                os.makedirs(temp_data_dir)
                os.makedirs(temp_model_dir)
                
                # Copy image to temp directory
                import shutil
                from PIL import Image
                
                # Get image basename
                img_basename = os.path.basename(image_path)
                temp_img_path = os.path.join(temp_data_dir, img_basename)
                
                # Copy and ensure it's .bin.png
                if not img_basename.endswith('.bin.png'):
                    temp_img_path = os.path.join(temp_data_dir, 
                                               os.path.splitext(img_basename)[0] + '.bin.png')
                
                shutil.copy2(image_path, temp_img_path)
                
                # Create corresponding .gt.txt file
                gt_path = temp_img_path.replace('.bin.png', '.gt.txt')
                with open(gt_path, 'w', encoding='utf-8') as f:
                    f.write(sGroundTruth)
                
                # Get prediction BEFORE training
                pred_before = self._predict_image(image_path)
                
                # Use continue learning for single case training
                from calamari_ocr.ocr.scenario import CalamariScenario
                from calamari_ocr.scripts.train import main as calamari_train
                
                # Setup training parameters for single epoch
                trainer_params = CalamariScenario.default_trainer_params()
                trainer_params.output_dir = temp_model_dir
                trainer_params.epochs = 1
                trainer_params.gen.train.images = [temp_img_path]
                trainer_params.gen.val.images = [temp_img_path]  # Use same for validation
                trainer_params.gen.setup.train.batch_size = 1
                trainer_params.gen.setup.train.num_processes = 1
                trainer_params.gen.setup.val.num_processes = 1
                trainer_params.progress_bar = False
                
                # Setup warmstart from existing model
                trainer_params.warmstart.model = self.checkpoint_path
                trainer_params.warmstart.allow_partial = True
                trainer_params.warmstart.trim_graph_name = False
                
                # Run training (this will update weights)
                start_time = time.time()
                result = calamari_train(trainer_params)
                elapsed = time.time() - start_time
                
                # Load the updated model for prediction
                updated_checkpoint = None
                for ckpt_name in ["best.ckpt.json", "best.ckpt"]:
                    ckpt_candidate = os.path.join(temp_model_dir, ckpt_name)
                    if os.path.exists(ckpt_candidate):
                        updated_checkpoint = ckpt_candidate
                        break
                
                if updated_checkpoint:
                    # Get prediction AFTER training
                    pred_after = self._predict_image_with_checkpoint(image_path, updated_checkpoint)
                    
                    # Calculate CER
                    cer_after = self._calculate_cer(pred_after, sGroundTruth)
                    
                    # Extract loss from result if available
                    loss = 0.0
                    try:
                        if hasattr(result, 'history') and result.history:
                            loss_history = result.history.get('loss', [])
                            if loss_history:
                                loss = float(loss_history[-1])
                    except:
                        pass
                    
                    return {
                        "loss": loss,
                        "pred": pred_after,
                        "cer": cer_after,
                        "pred_before": pred_before,
                        "training_time": elapsed,
                        "status": "ok"
                    }
                else:
                    return {
                        "loss": 0.0,
                        "pred": pred_before,
                        "cer": self._calculate_cer(pred_before, sGroundTruth),
                        "pred_before": pred_before,
                        "training_time": elapsed,
                        "status": "no_updated_model"
                    }
            
        except Exception as e:
            print(f"❌ Training step failed: {e}")
            return {
                "loss": 0.0,
                "pred": "ERROR",
                "cer": 1.0,
                "pred_before": "ERROR",
                "training_time": 0.0,
                "status": f"error: {str(e)}"
            }
    
    def _predict_image(self, image_path: str) -> str:
        """Predict using current loaded model"""
        try:
            from PIL import Image
            image = Image.open(image_path).convert('L')
            image_np = np.array(image)
            
            predictions = list(self.predictor.predict_raw([image_np]))
            if predictions and hasattr(predictions[0], "outputs"):
                outs = predictions[0].outputs
                if isinstance(outs, tuple) and len(outs) > 0:
                    pr_list = outs[0]
                    if isinstance(pr_list, list) and len(pr_list) > 0:
                        return pr_list[0].sentence
                if isinstance(outs, list) and len(outs) > 0:
                    best_guess = outs[0]
                    if hasattr(best_guess, "sentence"):
                        return best_guess.sentence
            return "ERROR"
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return "ERROR"
    
    def _predict_image_with_checkpoint(self, image_path: str, checkpoint_path: str) -> str:
        """Predict using specific checkpoint"""
        try:
            from calamari_ocr.ocr.predict.predictor import MultiPredictor
            from PIL import Image
            
            temp_predictor = MultiPredictor.from_paths(checkpoints=[checkpoint_path])
            image = Image.open(image_path).convert('L')
            image_np = np.array(image)
            
            predictions = list(temp_predictor.predict_raw([image_np]))
            if predictions and hasattr(predictions[0], "outputs"):
                outs = predictions[0].outputs
                if isinstance(outs, tuple) and len(outs) > 0:
                    pr_list = outs[0]
                    if isinstance(pr_list, list) and len(pr_list) > 0:
                        return pr_list[0].sentence
                if isinstance(outs, list) and len(outs) > 0:
                    best_guess = outs[0]
                    if hasattr(best_guess, "sentence"):
                        return best_guess.sentence
            return "ERROR"
        except Exception as e:
            print(f"❌ Prediction with checkpoint failed: {e}")
            return "ERROR"
    
    def _predict_with_engine(self, image_np: np.ndarray) -> str:
        """Predict using manual engine"""
        try:
            # Use the engine's internal model and codec for prediction
            prediction_raw = self.engine.oModel.predict(np.expand_dims(image_np, axis=0))
            return self.engine.oCodec.decode(prediction_raw[0])
        except Exception as e:
            print(f"❌ Manual engine prediction failed: {e}")
            return "ERROR"

    def _calculate_cer(self, sPred: str, sTarget: str) -> float:
        """Fast Edit Distance based CER calculation"""
        try:
            import editdistance
            return editdistance.eval(sPred, sTarget) / max(len(sTarget), 1)
        except ImportError:
            # Fallback: simple character-by-character comparison
            if len(sTarget) == 0:
                return 0.0 if len(sPred) == 0 else 1.0
    
    def save_model(self, output_path: str) -> bool:
        """Save the current model state"""
        try:
            if self.use_manual_engine and self.engine:
                # Use manual engine's save method
                self.engine.save_current_state(output_path)
                print(f"✅ Model saved using manual engine: {output_path}")
                return True
            else:
                # Fallback: use temporary training method for saving
                print(f"⚠️  Manual engine not available, using fallback save method")
                return self._save_with_temporary_method(output_path)
        except Exception as e:
            print(f"❌ Model saving failed: {e}")
            return False
    
    def _save_with_temporary_method(self, output_path: str) -> bool:
        """Fallback save method using temporary training"""
        try:
            # Create a minimal training setup just for saving
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_data_dir = os.path.join(temp_dir, "temp_data")
                os.makedirs(temp_data_dir)
                
                # Create a dummy image for saving
                from PIL import Image
                dummy_image = Image.new('L', (100, 50), color=255)
                dummy_path = os.path.join(temp_data_dir, "dummy.bin.png")
                dummy_image.save(dummy_path)
                
                # Create corresponding .gt.txt file
                gt_path = dummy_path.replace('.bin.png', '.gt.txt')
                with open(gt_path, 'w', encoding='utf-8') as f:
                    f.write("dummy")
                
                # Setup training parameters for saving
                from calamari_ocr.ocr.scenario import CalamariScenario
                from calamari_ocr.scripts.train import main as calamari_train
                
                trainer_params = CalamariScenario.default_trainer_params()
                trainer_params.output_dir = output_path
                trainer_params.epochs = 1
                trainer_params.gen.train.images = [dummy_path]
                trainer_params.gen.val.images = [dummy_path]
                trainer_params.gen.setup.train.batch_size = 1
                trainer_params.gen.setup.train.num_processes = 1
                trainer_params.gen.setup.val.num_processes = 1
                trainer_params.progress_bar = False
                
                # Setup warmstart from current checkpoint
                trainer_params.warmstart.model = self.checkpoint_path
                trainer_params.warmstart.allow_partial = True
                trainer_params.warmstart.trim_graph_name = False
                
                # Run training to save model
                result = calamari_train(trainer_params)
                
                if os.path.exists(output_path):
                    print(f"✅ Model saved using fallback method: {output_path}")
                    return True
                else:
                    print(f"❌ Fallback save failed")
                    return False
                    
        except Exception as e:
            print(f"❌ Fallback save failed: {e}")
            return False

    def request_stop(self):
        """Request graceful stop of training"""
        self.stop_requested = True
        if self.use_manual_engine and self.engine:
            self.engine.request_stop()

def print_separator(title: str):
    """Print a formatted separator"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def load_image_for_training(image_path: str) -> np.ndarray:
    """Load and preprocess image for training"""
    try:
        from PIL import Image
        image = Image.open(image_path).convert('L')
        return np.array(image)
    except Exception as e:
        print(f"❌ Failed to load image {image_path}: {e}")
        return None

def run_manual_training(data_folder: str, model_folder: str, case_id: Optional[int] = None, 
                       save_model: bool = False, output_model: Optional[str] = None) -> bool:
    """Run manual training with immediate weight updates"""
    
    print_separator("🎯 samuTrain Manual Training - Immediate Weight Updates")
    
    # Set environment variables
    os.environ["SAMUTRAIN_DATA_FOLDER"] = data_folder
    os.environ["SAMUTRAIN_MODEL_FOLDER"] = model_folder
    
    try:
        # Import modules after environment setup
        from db import Database
        from bridge import resolve_case_image_file
        
        print(f"📁 Data folder: {data_folder}")
        print(f"🤖 Model folder: {model_folder}")
        
        # Initialize database
        db = Database()
        
        # Verify inputs
        if not os.path.exists(data_folder):
            print(f"❌ Data folder not found: {data_folder}")
            return False
        
        if not os.path.exists(model_folder):
            print(f"❌ Model folder not found: {model_folder}")
            return False
        
        # Find checkpoint file
        checkpoint_path = None
        for ckpt_name in ["best.ckpt.json", "best.ckpt"]:
            ckpt_candidate = os.path.join(model_folder, ckpt_name)
            if os.path.exists(ckpt_candidate):
                checkpoint_path = ckpt_candidate
                break
        
        if not checkpoint_path:
            print(f"❌ No checkpoint found in {model_folder}")
            return False
        
        print(f"📋 Using checkpoint: {checkpoint_path}")
        
        # Get training cases
        training_cases = db.get_training_cases(include_gt_only=True)
        if not training_cases:
            print("❌ No training cases found in database")
            return False
        
        # Filter by case_id if specified
        if case_id is not None:
            training_cases = [c for c in training_cases if c['id'] == case_id]
            if not training_cases:
                print(f"❌ No case found with ID: {case_id}")
                return False
        
        print(f"📚 Found {len(training_cases)} training case(s) to process")
        
        # Initialize manual trainer
        trainer = CalamariManualTrainer(checkpoint_path)
        
        # Process each case
        print_separator("🎓 Starting Manual Training")
        total_processed = 0
        total_success = 0
        
        for i, case in enumerate(training_cases, 1):
            print(f"\n🔄 Processing Case {i}/{len(training_cases)}: {case['img_path']}")
            
            # Resolve image path
            image_path = resolve_case_image_file(case['img_path'])
            if not os.path.exists(image_path):
                print(f"❌ Image file not found: {image_path}")
                continue
            
            # Load image
            image = load_image_for_training(image_path)
            if image is None:
                continue
            
            gt_text = case.get('gt_text', '')
            if not gt_text:
                print(f"❌ No ground truth for case {case['id']}")
                continue
            
            print(f"   Ground Truth: '{gt_text}'")
            
            # Train on single case
            result = trainer.train_single_case(image_path, gt_text)
            
            if result['status'] == 'ok':
                total_success += 1
                print(f"   ✅ Training completed in {result['training_time']:.2f}s")
                print(f"   Before: '{result['pred_before']}'")
                print(f"   After:  '{result['pred']}'")
                print(f"   Loss:   {result['loss']:.4f}")
                print(f"   CER:    {result['cer']:.3f}")
                
                # Update database with new prediction
                try:
                    db.update_predictions_batch([{
                        'case_id': case['id'],
                        'ocr_text': result['pred'],
                        'confidence': 1.0 - result['cer']  # Simple confidence proxy
                    }])
                    print(f"   💾 Database updated")
                except Exception as e:
                    print(f"   ⚠️  Database update failed: {e}")
                
            else:
                print(f"   ❌ Training failed: {result['status']}")
                if result.get('pred_before'):
                    print(f"   Before: '{result['pred_before']}'")
            
            total_processed += 1
        
        print_separator("📊 Training Summary")
        print(f"Total cases processed: {total_processed}")
        print(f"Successful training:   {total_success}")
        print(f"Success rate:          {total_success/total_processed*100:.1f}%" if total_processed > 0 else "N/A")
        
        # Save model if requested
        if save_model and total_success > 0:
            try:
                if output_model is None:
                    # Generate automatic output path
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_model = f"{model_folder}_manual_trained_{timestamp}"
                
                print(f"\n💾 Saving updated model to: {output_model}")
                
                # Create a final training run to save the accumulated weights
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_data_dir = os.path.join(temp_dir, "temp_data")
                    temp_model_dir = os.path.join(temp_dir, "temp_model")
                    os.makedirs(temp_data_dir)
                    os.makedirs(temp_model_dir)
                    
                    # Use one of the successfully trained cases for final model save
                    successful_cases = [case for case in training_cases if any(
                        str(case['id']) in str(trainer.predictor) for _ in [1]
                    )]
                    
                    if successful_cases:
                        # Create a temporary dataset with one successful case
                        sample_case = successful_cases[0]
                        image_path = resolve_case_image_file(sample_case['img_path'])
                        gt_text = sample_case.get('gt_text', '')
                        
                        if os.path.exists(image_path) and gt_text:
                            # Setup final training for model saving
                            from calamari_ocr.ocr.scenario import CalamariScenario
                            from calamari_ocr.scripts.train import main as calamari_train
                            
                            # Copy image to temp directory
                            import shutil
                            img_basename = os.path.basename(image_path)
                            temp_img_path = os.path.join(temp_data_dir, img_basename)
                            if not img_basename.endswith('.bin.png'):
                                temp_img_path = os.path.join(temp_data_dir, 
                                                           os.path.splitext(img_basename)[0] + '.bin.png')
                            shutil.copy2(image_path, temp_img_path)
                            
                            # Create corresponding .gt.txt file
                            gt_path = temp_img_path.replace('.bin.png', '.gt.txt')
                            with open(gt_path, 'w', encoding='utf-8') as f:
                                f.write(gt_text)
                            
                            # Setup training parameters for final save
                            trainer_params = CalamariScenario.default_trainer_params()
                            trainer_params.output_dir = output_model
                            trainer_params.epochs = 1
                            trainer_params.gen.train.images = [temp_img_path]
                            trainer_params.gen.val.images = [temp_img_path]
                            trainer_params.gen.setup.train.batch_size = 1
                            trainer_params.gen.setup.train.num_processes = 1
                            trainer_params.gen.setup.val.num_processes = 1
                            trainer_params.progress_bar = False
                            
                            # Setup warmstart from the most recent checkpoint
                            trainer_params.warmstart.model = trainer.checkpoint_path
                            trainer_params.warmstart.allow_partial = True
                            trainer_params.warmstart.trim_graph_name = False
                            
                            # Run final training to save model
                            result = calamari_train(trainer_params)
                            
                            if os.path.exists(output_model):
                                print(f"✅ Model successfully saved to: {output_model}")
                                
                                # Update the trainer's checkpoint path for future use
                                new_checkpoint = None
                                for ckpt_name in ["best.ckpt.json", "best.ckpt"]:
                                    ckpt_candidate = os.path.join(output_model, ckpt_name)
                                    if os.path.exists(ckpt_candidate):
                                        new_checkpoint = ckpt_candidate
                                        break
                                
                                if new_checkpoint:
                                    trainer.checkpoint_path = new_checkpoint
                                    print(f"📋 New checkpoint: {new_checkpoint}")
                            else:
                                print(f"❌ Failed to save model to: {output_model}")
                        else:
                            print(f"⚠️  Could not create final model save - no valid case data")
                    else:
                        print(f"⚠️  No successful training cases to use for model save")
                        
            except Exception as e:
                print(f"❌ Model saving failed: {e}")
        
        return total_success > 0
        
    except Exception as e:
        print(f"❌ Manual training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point for manual training"""
    parser = argparse.ArgumentParser(
        description="samuTrain Manual Training - Immediate Weight Updates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python try_manual_training.py data/64_case models/generic_ocr_model_3
  python try_manual_training.py data/64_case models/generic_ocr_model_3 --case-id 1
  python try_manual_training.py data/single_case models/generic_ocr_model_3
        """
    )
    
    parser.add_argument("data_folder", help="Path to training data folder")
    parser.add_argument("model_folder", help="Path to existing model folder")
    parser.add_argument("--case-id", type=int, help="Train on specific case ID only")
    parser.add_argument("--save-model", action="store_true", help="Save updated model permanently after training")
    parser.add_argument("--output-model", help="Output path for saved model (requires --save-model)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.save_model and args.output_model and not os.path.isabs(args.output_model):
        args.output_model = os.path.abspath(args.output_model)
    
    # Convert to absolute paths
    data_folder = os.path.abspath(args.data_folder)
    model_folder = os.path.abspath(args.model_folder)
    
    print("🎯 samuTrain Manual Training")
    print("=" * 60)
    print("This script performs manual training with immediate weight updates")
    print("using train_on_batch() for granular control and real-time feedback.")
    print("=" * 60)
    
    success = run_manual_training(data_folder, model_folder, args.case_id, args.save_model, args.output_model)
    
    if success:
        print("\n🎉 Manual training completed successfully!")
        print("Model weights have been updated and database refreshed.")
        sys.exit(0)
    else:
        print("\n❌ Manual training failed!")
        print("Check the error messages above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
