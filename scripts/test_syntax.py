#!/usr/bin/env python3
"""
Simple syntax test for try_start_training.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    # Try to import the script as a module to check syntax
    import importlib.util
    spec = importlib.util.spec_from_file_location("try_start_training", os.path.join(project_root, "scripts", "try_start_training.py"))
    
    if spec is None:
        print("❌ Could not load module spec")
        sys.exit(1)
        
    module = importlib.util.module_from_spec(spec)
    
    # Try to execute the module (this will check syntax)
    print("🔍 Checking syntax...")
    spec.loader.exec_module(module)
    
    # Check if key functions exist
    required_functions = [
        'start_initial_training',
        'continue_learning', 
        'create_base_trainer_params',
        'create_trainer_params_for_initial_training',
        'setup_charset_extension',
        'setup_library_training_environment'
    ]
    
    missing_functions = []
    for func_name in required_functions:
        if not hasattr(module, func_name):
            missing_functions.append(func_name)
    
    if missing_functions:
        print(f"❌ Missing functions: {missing_functions}")
        sys.exit(1)
    
    print("✅ All required functions are present")
    print("✅ Syntax check passed")
    print("✅ Script is ready for use")
    
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    sys.exit(1)
except ImportError as e:
    print(f"⚠️  Import error (expected if Calamari not installed): {e}")
    print("✅ Syntax is correct, only missing optional dependencies")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
