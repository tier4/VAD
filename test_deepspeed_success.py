#!/usr/bin/env python
"""Test if DeepSpeed integration is working properly."""

import subprocess
import time
import signal
import sys

def test_deepspeed_training():
    """Run DeepSpeed training for a short time to verify it works."""
    print("Testing DeepSpeed integration...")
    
    # Start the training process
    cmd = ["./tools/dist_train_deepspeed.sh", "configs/VAD/VAD_base_e2e_deepspeed.py", "2"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Monitor output
    start_time = time.time()
    training_started = False
    errors = []
    
    try:
        for line in process.stdout:
            print(line.strip())
            
            # Check for successful indicators
            if "DeepSpeed optimizer: AdamW" in line:
                print("✓ DeepSpeed optimizer created successfully")
            elif "distributed groups summary" in line:
                print("✓ DeepSpeed distributed groups initialized")
            elif "DeepSpeed Flops Profiler Enabled" in line:
                print("✓ DeepSpeed engine initialized")
            elif "workflow:" in line and "max: 60 epochs" in line:
                print("✓ Training workflow started")
                training_started = True
            elif "Epoch [1]" in line or "iter:" in line:
                print("✓ Training iterations running")
                training_started = True
            
            # Check for errors
            if "RuntimeError" in line or "TypeError" in line:
                errors.append(line.strip())
            
            # Stop after 30 seconds or if training started
            if time.time() - start_time > 30:
                if training_started:
                    print("\n✓ SUCCESS: DeepSpeed training is working!")
                else:
                    print("\n⚠ WARNING: Training may not have started properly")
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    
    # Report results
    if errors:
        print("\nErrors encountered:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    return training_started and not errors

if __name__ == "__main__":
    success = test_deepspeed_training()
    sys.exit(0 if success else 1)