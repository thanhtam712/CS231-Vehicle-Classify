"""
Script để chạy prediction và evaluation cho tất cả các models
"""

import os
import glob
import subprocess
import sys


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    test_meta = os.path.join(project_root, 'vehicle-10', 'test_meta.json')
    dataset_dir = os.path.join(project_root, 'vehicle-10')
    models_dir = os.path.join(project_root, 'models')
    output_base = os.path.join(project_root, 'output')
    
    # Check if paths exist
    if not os.path.exists(test_meta):
        print(f"Error: test_meta.json not found at {test_meta}")
        return
    
    if not os.path.exists(dataset_dir):
        print(f"Error: dataset directory not found at {dataset_dir}")
        return
    
    if not os.path.exists(models_dir):
        print(f"Error: models directory not found at {models_dir}")
        return
    
    # Find all model files
    model_files = glob.glob(os.path.join(models_dir, '*.pkl'))
    
    if not model_files:
        print(f"Error: No .pkl model files found in {models_dir}")
        return
    
    print(f"Found {len(model_files)} model(s)")
    print(f"{'='*80}")
    
    # Evaluate each model
    for model_path in sorted(model_files):
        model_name = os.path.basename(model_path).replace('.pkl', '')
        output_dir = os.path.join(output_base, model_name)
        
        print(f"\nEvaluating model: {model_name}")
        print(f"{'-'*80}")
        
        # Run prediction script
        cmd = [
            sys.executable,
            os.path.join(script_dir, 'predict_test.py'),
            '--test_meta', test_meta,
            '--dataset_dir', dataset_dir,
            '--model', model_path,
            '--output_dir', output_dir
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {model_name}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("All evaluations complete!")
    print(f"Results saved in: {output_base}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
