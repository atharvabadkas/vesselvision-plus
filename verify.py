import os
import sys
import platform
import yaml
import torch
import getpass
from pathlib import Path
from ultralytics import YOLO
from typing import Optional, Dict, Union, Tuple, List, Any

# Import custom utilities
from modules.dataset_processing import load_dataset_config, verify_dataset_structure
from modules.hyperparameters import get_device, get_training_hyperparameters

def get_best_available_device():

    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

class EnvironmentVerifier:

    def __init__(self):
        self.version_info = {}
        self.checks_passed = {}
        self.warnings = []
        self.errors = []
        self.user_hardware = {}
        self.environment = os.environ.copy()
        self.wandb_project = "vessel_detection"
    
    def verify_all(self) -> bool:

        self.check_python_version()
        self.check_pytorch_available()
        self.check_cuda_available()
        self.check_libraries()
        self.check_dataset_files()
        
        print("\n=== Environment Verification Results ===")
        
        for check_name, passed in self.checks_passed.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{check_name:<30} {status}")
        
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"- {warning}")
        
        all_required_passed = all([v for k, v in self.checks_passed.items() if k not in ['CUDA Available']])
        
        print("\nVerification " + ("PASSED" if all_required_passed else "FAILED"))
        
        if not all_required_passed:
            print("\nPlease address the issues above before running train.py")
            
        return all_required_passed
        
    def print_system_info(self) -> None:

        print("\n=== System Information ===")
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        
        if self.version_info.get('pytorch'):
            print(f"PyTorch: {self.version_info['pytorch']}")
        
        if self.version_info.get('ultralytics'):
            print(f"Ultralytics: {self.version_info['ultralytics']}")
            
        if self.user_hardware.get('gpu'):
            print(f"GPU: {self.user_hardware['gpu']}")
            if self.user_hardware.get('cuda_version'):
                print(f"CUDA Version: {self.user_hardware['cuda_version']}")
        else:
            print("GPU: None detected")
    
    def check_python_version(self):
        # Implementation of check_python_version method
        pass

    def check_pytorch_available(self):
        # Implementation of check_pytorch_available method
        pass

    def check_cuda_available(self):
        try:
            cuda_available = torch.cuda.is_available()
            self.checks_passed['CUDA Available'] = cuda_available
            
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    gpu_names = []
                    for i in range(gpu_count):
                        gpu_names.append(torch.cuda.get_device_name(i))
                    
                    self.user_hardware['gpu'] = ", ".join(gpu_names)
                    self.user_hardware['cuda_version'] = torch.version.cuda
                    print(f"Found {gpu_count} GPU(s): {self.user_hardware['gpu']}")
                else:
                    self.warnings.append("CUDA is available but no GPU devices found")
                    self.user_hardware['gpu'] = "None detected (unexpected)"
            else:
                self.warnings.append("No GPU detected - training will be VERY slow on CPU only")
                self.user_hardware['gpu'] = None
                
                # Check if this is a Mac with Apple Silicon
                if platform.system() == "Darwin" and platform.machine() == "arm64":
                    # Check if MPS is available (Metal Performance Shaders for Mac M1/M2/M3)
                    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                        self.user_hardware['gpu'] = "Apple Silicon (MPS)"
                        self.warnings.append("Apple Silicon detected - will use MPS for acceleration")
                    else:
                        self.warnings.append("Apple Silicon detected but MPS is not available - consider installing PyTorch with MPS support")
        except Exception as e:
            self.warnings.append(f"Error checking CUDA: {str(e)}")
            self.checks_passed['CUDA Available'] = False
            self.user_hardware['gpu'] = None

    def check_libraries(self):
        # Implementation of check_libraries method
        pass

    def check_dataset_files(self):
        # Implementation of check_dataset_files method
        pass

def setup_environment():
    print("\nSetting up environment...")
    
    verifier = EnvironmentVerifier()
    all_passed = verifier.verify_all()
    verifier.print_system_info()
    
    if not all_passed:
        print("\nWARNING: Some checks failed. The train.py script may still work but could encounter issues.")
        print("Do you want to continue anyway? (y/n)")
        choice = input().lower()
        if choice != 'y':
            sys.exit(1)
    
    return verifier

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Environment verification for the vessel detection project')
    parser.add_argument('--setup', action='store_true', help='Setup a clean environment')
    parser.add_argument('--info', action='store_true', help='Print system information only')
    
    args = parser.parse_args()
    
    if args.info:
        verifier = EnvironmentVerifier()
        verifier.print_system_info()
        sys.exit(0)
    
    if args.setup:
        setup_environment()
    else:
        # Just run verification by default
        verifier = EnvironmentVerifier()
        verifier.verify_all()
        verifier.print_system_info() 