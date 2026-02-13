"""Codec file for Calamari training."""

def main():
    """Main function for codec."""
    import os
    from calamari_ocr.ocr.dataset.datareader.file import FileDataWriter
    
    # Create codec data
    codec_data = []
    
    # Add training files
    files = [
        "data/test_data/test_single/test001.bin.png"
    ]
    
    for file_path in files:
        gt_path = file_path.replace('.bin.png', '.gt.txt')
        codec_data.append({
            'image_path': file_path,
            'text_path': gt_path
        })
    
    # Write codec file
    writer = FileDataWriter()
    writer.write(codec_data, 'train_codec.json')

if __name__ == "__main__":
    main()
