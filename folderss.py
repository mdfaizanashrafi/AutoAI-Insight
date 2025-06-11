import os
'''
project_root = 'autoai-insight'

# Create directory structure
folders = [
    'backend/data_pipeline',
    'backend/eda',
    'backend/supervised_models',
    'backend/evaluation',
    'backend/utils',
    'data/raw',
    'data/processed',
    'models',
    'reports',
    'logs',
    'config'
]

for folder in folders:
    os.makedirs(os.path.join(project_root, folder), exist_ok=True)

print("✅ Folder structure created.")

modules=[
    'backend/data_pipeline/__init__.py',
    'backend/eda/__init__.py',
    'backend/supervised_models/__init__.py',
    'backend/evaluation/__init__.py',
    'backend/utils/__init__.py'
]
for module in modules:
    open (os.path.join(project_root,module),'a').close()

print ("Init FIles created")


# test_imports.py

try:
    from backend.data_pipeline.data_loader import load_data
    from backend.eda.eda_analyzer import EDAAnalyzer
    from backend.eda.plot_generator import PlotGenerator
    print("✅ All imports worked!")
except Exception as e:
    print("❌ Import failed:", e)
'''

# test_import.py

import sys
sys.path.append(".")  # Add current folder to Python path

try:
    from backend.data_pipeline.utils import get_file_extension
    print("✅ Import successful!")
except Exception as e:
    print("❌ Import failed:", e)