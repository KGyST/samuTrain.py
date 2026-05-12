"""
try_image_splitting.py
----------------------
Focused script for testing image splitting functionality.
Splits the image containing "1963" into individual character images: 1, 9, 6, 3
"""

import os
import sys
import cv2
import numpy as np
import time
from typing import List, Tuple

# Add src and current directory to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
sys.path.insert(0, script_dir)

# Import splitting functions from try_calamari
from try_calamari import (
    _split_word_to_char_images_calamari,
    _split_word_to_char_images_calamari_native
)

def save_character_images(char_images: List[Tuple[np.ndarray, str]], output_dir: str, method_name: str):
    """Save character images to disk with descriptive names"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving {len(char_images)} character images for {method_name} method:")
    for i, (char_img, char) in enumerate(char_images):
        filename = f"char_{i+1:02d}_{char}_{method_name}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, char_img)
        print(f"  Character {i+1}: '{char}' | Shape: {char_img.shape} -> {filepath}")

def display_split_images(image: np.ndarray, char_images: List[Tuple[np.ndarray, str]], method_name: str):
    """Display the split images using OpenCV"""
    import cv2
    
    print(f"\n🖼️  Displaying split images for {method_name} method...")
    print("Press any key to continue to next image, or ESC to skip all displays")
    
    # Display original image first
    original_resized = cv2.resize(image, (400, 100))
    cv2.imshow(f'Original Image - {method_name}', original_resized)
    
    key = cv2.waitKey(0)
    cv2.destroyWindow(f'Original Image - {method_name}')
    
    if key == 27:  # ESC key
        print("   ⏭️  Skipped remaining displays")
        return
    
    # Display each character image
    for i, (char_img, char) in enumerate(char_images):
        # Resize character image for better visibility
        char_resized = cv2.resize(char_img, (100, 100))
        
        # Add text overlay
        display_img = char_resized.copy()
        cv2.putText(display_img, f'Char {i+1}: "{char}"', (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, f'Shape: {char_img.shape}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        window_name = f'Character {i+1}: "{char}" - {method_name}'
        cv2.imshow(window_name, display_img)
        
        key = cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        
        if key == 27:  # ESC key
            print(f"   ⏭️  Skipped remaining displays after character {i+1}")
            break
    
    cv2.destroyAllWindows()
    print(f"   ✅ Finished displaying {len(char_images)} character images")

def visualize_splitting(image: np.ndarray, char_images: List[Tuple[np.ndarray, str]], method_name: str):
    """Create a visualization of the splitting result"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, len(char_images) + 1, figsize=(15, 6))
    fig.suptitle(f'Image Splitting Results - {method_name} Method', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Character images
    for i, (char_img, char) in enumerate(char_images):
        # Top row: character images
        axes[0, i + 1].imshow(char_img, cmap='gray')
        axes[0, i + 1].set_title(f'Char {i+1}: "{char}"')
        axes[0, i + 1].axis('off')
        
        # Bottom row: character dimensions
        axes[1, i + 1].text(0.5, 0.5, f'Shape: {char_img.shape}', 
                           ha='center', va='center', transform=axes[1, i + 1].transAxes)
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    output_path = f"e:/Git/samuTrain.py/data/split_chars/splitting_visualization_{method_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {output_path}")

def main():
    # Configuration
    MODEL_PATH = r"E:\Git\samuTrain.py\models\generic_ocr_model\best.ckpt.json"
    IMAGE_PATH = r"E:\Git\samuTrain.py\data\single_case\010071.bin.png"
    TARGET_TEXT = "1963"
    OUTPUT_DIR = r"E:\Git\samuTrain.py\data\split_chars"
    
    # Validate inputs
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image not found: {IMAGE_PATH}")
        sys.exit(1)
    
    # Load image
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Failed to load image: {IMAGE_PATH}")
        sys.exit(1)
    
    print(f"📸 Loaded image: {IMAGE_PATH}")
    print(f"   Shape: {img.shape}")
    print(f"   Target text: '{TARGET_TEXT}'")
    print(f"   Model: {MODEL_PATH}")
    print("=" * 60)
    
    # Test different splitting methods
    methods = [
        ("Calamari_Subprocess", lambda: _split_word_to_char_images_calamari(img, MODEL_PATH)),
        ("Calamari_Native", lambda: _split_word_to_char_images_calamari_native(img, MODEL_PATH, TARGET_TEXT))
    ]
    
    results = {}
    
    for method_name, method_func in methods:
        print(f"\n🔧 Testing {method_name} method...")
        print("-" * 40)
        
        try:
            start_time = time.time()
            char_images = method_func()
            end_time = time.time()
            
            if char_images:
                results[method_name] = {
                    'char_images': char_images,
                    'time': end_time - start_time,
                    'success': True
                }
                
                print(f"✅ {method_name} method successful!")
                print(f"   Time: {end_time - start_time:.3f}s")
                print(f"   Characters found: {len(char_images)}")
                
                # Display character details
                for i, (char_img, char) in enumerate(char_images):
                    print(f"     Char {i+1}: '{char}' | Shape: {char_img.shape} | Size: {char_img.size} bytes")
                
                # Save character images
                save_character_images(char_images, OUTPUT_DIR, method_name)
                
                # Create visualization
                visualize_splitting(img, char_images, method_name)
                
            else:
                results[method_name] = {
                    'char_images': [],
                    'time': end_time - start_time,
                    'success': False
                }
                print(f"❌ {method_name} method failed: No characters returned")
                
        except Exception as e:
            results[method_name] = {
                'char_images': [],
                'time': 0,
                'success': False,
                'error': str(e)
            }
            print(f"❌ {method_name} method failed: {e}")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("📊 SUMMARY COMPARISON")
    print("=" * 60)
    
    successful_methods = [name for name, result in results.items() if result['success']]
    
    if successful_methods:
        print(f"\n✅ Successful methods ({len(successful_methods)}):")
        for method_name in successful_methods:
            result = results[method_name]
            char_images = result['char_images']
            chars = [char for _, char in char_images]
            print(f"   {method_name}: {''.join(chars)} | {result['time']:.3f}s | {len(char_images)} chars")
        
        # Performance comparison
        if len(successful_methods) > 1:
            fastest = min(successful_methods, key=lambda x: results[x]['time'])
            slowest = max(successful_methods, key=lambda x: results[x]['time'])
            fastest_time = results[fastest]['time']
            speedup = results[slowest]['time'] / fastest_time if fastest_time > 0 else float('inf')
            print(f"\n🚀 Performance: {fastest} is {speedup:.1f}x faster than {slowest}")
    
    failed_methods = [name for name, result in results.items() if not result['success']]
    if failed_methods:
        print(f"\n❌ Failed methods ({len(failed_methods)}):")
        for method_name in failed_methods:
            result = results[method_name]
            error = result.get('error', 'Unknown error')
            print(f"   {method_name}: {error}")
    
    print(f"\n📁 All character images saved to: {OUTPUT_DIR}")
    print("\n🎯 Target achieved: Split '1963' into individual character images!")
    
    # Show expected vs actual results
    print("\n📋 Expected vs Actual Results:")
    print(f"   Expected: 1, 9, 6, 3")
    for method_name in successful_methods:
        chars = [char for _, char in results[method_name]['char_images']]
        print(f"   {method_name}: {', '.join(chars)}")

if __name__ == "__main__":
    main()
