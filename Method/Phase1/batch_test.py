import yaml
import os
import argparse
import copy
import torch.multiprocessing as mp
import time
import glob
import datetime
import traceback 
import torch 
from queue import Empty

from Phase1.ImageEOG_init_universal import UniversalPatchEOGenAttack
from models.model_factory import get_model_wrapper
from utils.image_utils import load_image_for_art

# === Task List ===
MODELS_TO_TEST = [
    {"name": "blip-large", "path": "models/blip-image-captioning-large", "batch_size": 100}
    # {"name": "llava-v1.5-7b", "path": "models/llava-hf-v1.5-7b", "batch_size": 100}
    # {"name": "qwen3-vl-8b", "path": "models/Qwen3-VL-8B-Instruct", "batch_size": 100}
    # {"name": "internvl3", "path": "models/InternVL3-8B", "batch_size": 100}
    # {"name": "qwen3-vl-2b", "path": "models/Qwen3-VL-2B-Instruct", "batch_size": 100}
    # {"name": "qwen3-vl-4b", "path": "models/Qwen3-VL-4B-Instruct", "batch_size": 100}
    # {"name": "qwen3-vl-32b", "path": "models/Qwen3-VL-32B-Instruct", "batch_size": 100}
]


def worker_process(gpu_id, task_queue, base_config, run_id):
    """
    Worker process for evaluating models on designated GPUs.
    """
    print(f"[GPU {gpu_id}] Worker started. Run ID: {run_id}")
    
    dataset_dir = base_config['data'].get('dataset_dir', "data/dataset_clean")
    
    image_files = []
    if os.path.exists(dataset_dir):
        raw_files = sorted(glob.glob(os.path.join(dataset_dir, "*")))
        image_files = [f for f in raw_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    
    if not image_files:
        print(f"[GPU {gpu_id}] Warning: Dataset empty. Using default input_image.")
        image_files = [base_config['data']['input_image']]
    
    print(f"[GPU {gpu_id}] Found {len(image_files)} images.")

    while True:
        try:
            model_info = task_queue.get(timeout=3)
        except Empty:
            break
            
        print(f"[GPU {gpu_id}] Starting model: {model_info['name']}")
        
        current_config = copy.deepcopy(base_config)
        current_config['model']['name'] = model_info['name']
        current_config['model']['path'] = model_info['path']
        current_config['model']['batch_size'] = model_info['batch_size']
        current_config['model']['device'] = f"cuda:{gpu_id}"
        
        gpu_output_root = os.path.join(
            current_config['data']['output_dir'], 
            run_id, 
            f"gpu_{gpu_id}_{model_info['name']}"
        )
        current_config['data']['output_dir'] = gpu_output_root
        
        try:
            # Load model
            wrapper = get_model_wrapper(current_config)
            
            for img_idx, img_path in enumerate(image_files):
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Resume from checkpoint
                task_dir = os.path.join(gpu_output_root, img_name)
                marker_file = os.path.join(task_dir, "attack_progress.csv")
                
                if os.path.exists(marker_file):
                    print(f"[GPU {gpu_id}] Skipping {img_name} (Done).")
                    continue

                print(f"[GPU {gpu_id}] Processing {img_name} ({img_idx+1}/{len(image_files)})...")
                
                try:
                    clean_img = load_image_for_art(img_path)
                    attacker = UniversalPatchEOGenAttack(wrapper, current_config, img_id=img_name)
                    attacker.run(clean_img)
                    print(f"[GPU {gpu_id}] {img_name} Done.")
                    
                except Exception as e:
                    print(f"[GPU {gpu_id}] Error on {img_name}: {e}")
                    traceback.print_exc()
                    continue
            
            print(f"[GPU {gpu_id}] Finished task {model_info['name']}")
            
            # Cleanup
            del wrapper
            torch.cuda.empty_cache() 
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Critical Model Init Error: {e}")
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="configs/configs.yaml")
    parser.add_argument("--resume_id", type=str, default=None)
    args = parser.parse_args()
    
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)

    if args.resume_id:
        run_id = args.resume_id
        print(f"Resuming Run ID: {run_id}")
    else:
        run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        print(f"Starting New Run ID: {run_id}")

    task_queue = mp.Queue()
    for model in MODELS_TO_TEST:
        task_queue.put(model)
        
    num_gpus = 4 
    processes = []
    
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, task_queue, base_config, run_id))
        p.start()
        processes.append(p)
        time.sleep(2)
        
    for p in processes:
        p.join()
        
    print(f"All tasks finished.")

if __name__ == "__main__":
    main()
