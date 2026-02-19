#!/usr/bin/env python3

import os
import calamari_ocr

print("calamari_ocr.__file__:", calamari_ocr.__file__)

# Confirm shadowing: check if it's the local lib/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
expected_path = os.path.join(project_root, 'lib', 'calamari', 'calamari_ocr', '__init__.py')

if calamari_ocr.__file__ == expected_path:
    print("SUCCESS: Shadowing is working - using local editable install from lib/")
else:
    print("WARNING: Not using local lib/, possibly not shadowed. Path:", calamari_ocr.__file__)
