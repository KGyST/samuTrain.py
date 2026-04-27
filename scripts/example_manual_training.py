#!/usr/bin/env python3
# Example usage of the manual training script
# This demonstrates how to use the CalamariManualTrainer for single-case training

import os
import sys

# Add src/ to sys.path
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectRoot = os.path.dirname(scriptDir)
srcDir = os.path.join(projectRoot, 'src')
sys.path.insert(0, srcDir)

def example_single_case_training():
    """Example of training on a single case for 10 epochs with model saving"""
    from try_manual_training import CalamariManualTrainer
    import tempfile
    import shutil
    
    # Initialize trainer
    checkpoint_path = os.path.join(projectRoot, "models/generic_ocr_model/best.ckpt.json")
    trainer = CalamariManualTrainer(checkpoint_path)
    
    # Train on a single case
    image_path = os.path.join(projectRoot, "data/single_case/01002c.bin.png")
    ground_truth = "1962"
    
    print(f"🎯 Multi-Epoch Manual Training Example")
    print("=" * 50)
    print(f"Training on: {image_path}")
    print(f"Ground truth: '{ground_truth}'")
    print(f"Total epochs: 10")
    print("=" * 50)
    
    # Track training progress
    initial_prediction = None
    final_prediction = None
    total_time = 0
    cer_history = []
    loss_history = []
    
    # Training loop for 10 epochs
    for epoch in range(1, 11):
        print(f"\n📚 Epoch {epoch}/10")
        print("-" * 30)
        
        # Train on the same case
        result = trainer.train_single_case(image_path, ground_truth)
        
        if result['status'] == 'ok':
            # Store initial prediction (before first training)
            if epoch == 1:
                initial_prediction = result['pred_before']
            
            # Update final prediction (after each training)
            final_prediction = result['pred']
            
            # Track metrics
            cer_history.append(result['cer'])
            loss_history.append(result['loss'])
            total_time += result['training_time']
            
            print(f"   Before: '{result['pred_before']}'")
            print(f"   After:  '{result['pred']}'")
            print(f"   CER:    {result['cer']:.3f}")
            print(f"   Loss:   {result['loss']:.4f}")
            print(f"   Time:   {result['training_time']:.2f}s")
            
            # Show improvement
            if epoch > 1:
                cer_improvement = cer_history[-2] - result['cer']
                if cer_improvement > 0:
                    print(f"   📈 CER improved by {cer_improvement:.3f}")
                else:
                    print(f"   📉 CER changed by {cer_improvement:.3f}")
        else:
            print(f"   ❌ Training failed: {result['status']}")
            break
    
    # Save the final model
    print(f"\n💾 Saving Final Model")
    print("=" * 50)
    
    try:
        # Create output model path
        output_model_path = os.path.join(projectRoot, "models", "example_multi_epoch_trained")
        
        # Use the same model saving logic as the main script
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_data_dir = os.path.join(temp_dir, "temp_data")
            temp_model_dir = os.path.join(temp_dir, "temp_model")
            os.makedirs(temp_data_dir)
            os.makedirs(temp_model_dir)
            
            # Copy image to temp directory for final model save
            img_basename = os.path.basename(image_path)
            temp_img_path = os.path.join(temp_data_dir, img_basename)
            if not img_basename.endswith('.bin.png'):
                temp_img_path = os.path.join(temp_data_dir, 
                                           os.path.splitext(img_basename)[0] + '.bin.png')
            shutil.copy2(image_path, temp_img_path)
            
            # Create corresponding .gt.txt file
            gt_path = temp_img_path.replace('.bin.png', '.gt.txt')
            with open(gt_path, 'w', encoding='utf-8') as f:
                f.write(ground_truth)
            
            # Setup training parameters for final save
            from calamari_ocr.ocr.scenario import CalamariScenario
            from calamari_ocr.scripts.train import main as calamari_train
            
            trainer_params = CalamariScenario.default_trainer_params()
            trainer_params.output_dir = output_model_path
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
            print(f"   🔄 Exporting model to: {output_model_path}")
            result = calamari_train(trainer_params)
            
            if os.path.exists(output_model_path):
                print(f"   ✅ Model successfully saved!")
                
                # Find the new checkpoint
                new_checkpoint = None
                for ckpt_name in ["best.ckpt.json", "best.ckpt"]:
                    ckpt_candidate = os.path.join(output_model_path, ckpt_name)
                    if os.path.exists(ckpt_candidate):
                        new_checkpoint = ckpt_candidate
                        break
                
                if new_checkpoint:
                    print(f"   📋 New checkpoint: {new_checkpoint}")
            else:
                print(f"   ❌ Failed to save model")
                
    except Exception as e:
        print(f"   ❌ Model saving failed: {e}")
    
    # Final summary
    print(f"\n📊 Training Summary")
    print("=" * 50)
    print(f"Total epochs completed: {len(cer_history)}")
    print(f"Total training time:   {total_time:.2f}s")
    print(f"Average time per epoch: {total_time/len(cer_history):.2f}s")
    
    if cer_history:
        print(f"Initial CER: {cer_history[0]:.3f}")
        print(f"Final CER:   {cer_history[-1]:.3f}")
        cer_improvement = cer_history[0] - cer_history[-1]
        print(f"CER improvement: {cer_improvement:.3f}")
        
        if loss_history:
            print(f"Initial Loss: {loss_history[0]:.4f}")
            print(f"Final Loss:   {loss_history[-1]:.4f}")
            loss_improvement = loss_history[0] - loss_history[-1]
            print(f"Loss improvement: {loss_improvement:.4f}")
    
    print(f"Initial prediction: '{initial_prediction}'")
    print(f"Final prediction:   '{final_prediction}'")
    print(f"Ground truth:       '{ground_truth}'")

if __name__ == "__main__":
    example_single_case_training()
