import os
import argparse
import yaml
import glob
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import time
from PIL import Image

from models.model_factory import get_model_wrapper
from utils.image_utils import load_image_for_art, save_adversarial_image
from utils.metrics import calculate_l2_distance
from attacks.hsja_masked_2 import MaskedHopSkipJump
from attacks.hsja_original_traced import OriginalHopSkipJump

# === GPU Task Configuration ===
GPU_CONFIG = {
    0: {
        "model_name": "blip-large", 
        "path": "models/blip-image-captioning-large", 
        "target_folder_keyword": "blip", 
        "batch_size": 128
    },
    1: {
        "model_name": "llava-v1.5-7b", 
        "path": "models/llava-hf-v1.5-7b",
        "target_folder_keyword": "llava",
        "batch_size": 128 
    },
    2: {
        "model_name": "qwen3-vl-8b", 
        "path": "models/Qwen3-VL-8B-Instruct",
        "target_folder_keyword": "qwen",
        "batch_size": 64 
    },
    3: {
       "model_name": "internvl3", 
       "path": "models/InternVL3-8B",
       "target_folder_keyword": "internvl",
       "batch_size": 64
    }
}

def find_best_rupe_files(img_dir):
    best_imgs = glob.glob(os.path.join(img_dir, "rupe_best_*.png"))
    if not best_imgs:
        return None, None
    try:
        best_imgs.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        target_img = best_imgs[-1]
        gen_id = target_img.split('_')[-1].split('.')[0]
        target_mask = os.path.join(img_dir, f"rupe_mask_{gen_id}.png")
        if os.path.exists(target_mask):
            return target_img, target_mask
        else:
            return target_img, None 
    except:
        return None, None

def worker_process(gpu_id, config_entry, base_config, phase1_root, phase2_root):
    print(f"[GPU {gpu_id}] Phase 2 Worker Started for {config_entry['model_name']}")
    
    current_config = base_config.copy()
    current_config['model']['name'] = config_entry['model_name']
    current_config['model']['path'] = config_entry['path']
    current_config['model']['batch_size'] = config_entry['batch_size']
    current_config['model']['device'] = f"cuda:{gpu_id}"
    
    search_pattern = os.path.join(phase1_root, f"*{config_entry['target_folder_keyword']}*")
    candidates = glob.glob(search_pattern)
    
    if not candidates:
        print(f"[GPU {gpu_id}] No Phase 1 results found.")
        return
        
    model_phase1_dir = candidates[0] 
    img_folders = [f for f in glob.glob(os.path.join(model_phase1_dir, "*")) if os.path.isdir(f)]
    img_folders.sort()
    
    try:
        wrapper = get_model_wrapper(current_config)
    except Exception as e:
        print(f"[GPU {gpu_id}] Model Init Error: {e}")
        return

    for img_folder in img_folders:
        img_name = os.path.basename(img_folder)
        output_dir = os.path.join(phase2_root, os.path.basename(model_phase1_dir), img_name)
        os.makedirs(output_dir, exist_ok=True)
        
        log_csv_path = os.path.join(output_dir, "log_trace.csv")
            
        print(f"[GPU {gpu_id}] Target: {img_name}")
        
        rupe_img_path, rupe_mask_path = find_best_rupe_files(img_folder)
        if not rupe_img_path or not rupe_mask_path:
            continue
            
        dataset_dir = base_config['data'].get('dataset_dir', 'data/dataset_clean')
        clean_candidates = glob.glob(os.path.join(dataset_dir, f"{img_name}.*"))
        if not clean_candidates:
            continue
        clean_img_path = clean_candidates[0]
        
        try:
            x_clean_3d = load_image_for_art(clean_img_path) 
            x_init_3d = load_image_for_art(rupe_img_path)
            
            mask_pil = Image.open(rupe_mask_path).convert('RGB')
            mask_3d = np.array(mask_pil).transpose(2, 0, 1) / 255.0
            mask_3d = (mask_3d > 0.5).astype(np.float32) 
            
            x_clean_batch = x_clean_3d[np.newaxis, ...]
            x_init_batch = x_init_3d[np.newaxis, ...]
            mask_batch = mask_3d[np.newaxis, ...]
            
            clean_len = wrapper.get_clean_length(x_clean_batch)
            target_threshold = int(clean_len * base_config['attack']['target_threshold_ratio'])
            init_len = wrapper.get_clean_length(x_init_batch)
            
            print(f"[GPU {gpu_id}] Base: {clean_len}, Target: >{target_threshold}, Init: {init_len}")
            
            if init_len <= target_threshold:
                print(f"[GPU {gpu_id}] Phase 1 Failed. Skipping.")
                continue
                
            attack_config = base_config['attack']
            wrapper.set_threshold(target_threshold)
            trace_csv_path = os.path.join(output_dir, "hsja_trace.csv")

            target_l2 = attack_config.get('l2_target_threshold', 10.0)
            
            attacker = MaskedHopSkipJump(
                classifier=wrapper,
                trace_log_path=trace_csv_path,
                l2_target_threshold=target_l2, 
                targeted=True,
                max_iter=attack_config['max_iter'],
                max_eval=attack_config['max_eval'],
                init_eval=attack_config['init_eval'],
                init_size=10, 
                norm=2,
                batch_size=attack_config['batch_eval'],
                verbose=False
            )

            # attacker = OriginalHopSkipJump(
            #     classifier=wrapper,
            #     trace_log_path=trace_csv_path,
            #     targeted=True,
            #     max_iter=attack_config['max_iter'],
            #     max_eval=attack_config['max_eval'],
            #     init_eval=attack_config['init_eval'],
            #     init_size=10, 
            #     norm=2,
            #     batch_size=attack_config['batch_eval'],
            #     verbose=False
            # )
            
            start_time = time.time()
            target_y = np.array([[0, 1]], dtype=np.float32)
            
            x_adv = attacker.generate(
                x=x_clean_batch,
                y=target_y,
                x_adv_init=x_init_batch,
                mask=mask_batch,
                resume=False
            )
            
            elapsed_time = time.time() - start_time
            
            # --- Dimensionality Processing ---
            if x_adv.ndim == 3:
                x_adv_batch = x_adv[np.newaxis, ...]  
            elif x_adv.ndim == 4:
                x_adv_batch = x_adv                   
            else:
                continue 
                
            x_adv_single = x_adv_batch[0]
            
            # === [VERIFICATION BLOCK START] ===
            print(f"\n[GPU {gpu_id}] === Verification Report for {img_name} ===")
            
            # 1. Quantization Verification
            len_float = wrapper.get_clean_length(x_adv_batch)
            
            save_path = os.path.join(output_dir, "hsja_result.png")
            save_adversarial_image(x_adv_single, save_path) # Save
            
            x_loaded = load_image_for_art(save_path) # Reload as if from disk
            x_loaded_batch = x_loaded[np.newaxis, ...]
            len_uint8 = wrapper.get_clean_length(x_loaded_batch)
            
            quant_status = "PASS" if len_uint8 > target_threshold else "FAIL (Quantization Loss)"
            print(f"[Quantization] Float32 Len: {len_float} | PNG Loaded Len: {len_uint8} | Status: {quant_status}")
            
            # 2. Batch Consistency Verification
            len_single = len_float

            x_batch_10 = np.tile(x_adv_single, (10, 1, 1, 1))
            try:
                inputs_10 = wrapper._process_inputs(x_batch_10)
                with torch.no_grad():
                    output_ids_10 = wrapper.generate(inputs_10)
                # Extract the length of the first sample
                len_in_batch = wrapper.get_token_length(output_ids_10[0])
                
                batch_status = "PASS" if len_in_batch == len_single else "FAIL (Batch Inconsistency)"
                print(f"[Batch Check]  Single Len: {len_single} | In-Batch(10) Len: {len_in_batch} | Status: {batch_status}")
                
            except Exception as e:
                print(f"[Batch Check]  Error: {e}")

            print("==================================================\n")
            # === [VERIFICATION BLOCK END] ===

            final_len = len_uint8 # Use the realistic one
            init_l2 = calculate_l2_distance(x_clean_3d, x_init_3d)
            final_l2 = calculate_l2_distance(x_clean_3d, x_adv_single)
            
            df = pd.DataFrame([{
                'CleanLen': clean_len,
                'TargetThreshold': target_threshold,
                'InitLen': init_len,
                'InitL2': init_l2,
                'FinalLen': final_len,
                'FinalL2': final_l2,
                'Queries': attacker.total_queries,
                'Time': elapsed_time,
                'FloatLen': len_float, 
                'QuantFail': (len_float > target_threshold and len_uint8 <= target_threshold)
            }])
            df.to_csv(log_csv_path, index=False)
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {img_name}: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1_run_id", type=str, required=True, help="Run ID of Phase 1")
    parser.add_argument("--base_config", type=str, default="configs/configs.yaml")
    args = parser.parse_args()
    
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)
        
    phase1_root = os.path.join(base_config['data']['output_dir'], args.phase1_run_id)
    phase2_root = os.path.join(base_config['data']['output_dir'], "phase_2_256_original", args.phase1_run_id)
    os.makedirs(phase2_root, exist_ok=True)
    
    processes = []
    for gpu_id, config_entry in GPU_CONFIG.items():
        p = mp.Process(
            target=worker_process, 
            args=(gpu_id, config_entry, base_config, phase1_root, phase2_root)
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
