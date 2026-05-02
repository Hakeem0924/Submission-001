import os
import argparse
import numpy as np
import torch
import time
import pandas as pd
import io
from PIL import Image, ImageFilter
import torchvision.transforms.functional as F
import yaml
from tqdm import tqdm
import glob

from models.model_factory import get_model_wrapper
from utils.image_utils import load_image_for_art

# ==========================================
# 0. Global Determinism (Reproducibility)
# ==========================================
def set_global_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# ==========================================
# 1. Classical Defense Implementations (Defense Transformations)
# ==========================================

def apply_jpeg_compression(image_np, quality):
    img_hwc = np.transpose(image_np, (1, 2, 0))
    img_uint8 = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    
    jpeg_pil = Image.open(buffer).convert("RGB")
    jpeg_np = np.array(jpeg_pil).astype(np.float32) / 255.0
    jpeg_chw = np.transpose(jpeg_np, (2, 0, 1))
    
    return jpeg_chw

def apply_gaussian_blur(image_np, radius):
    img_hwc = np.transpose(image_np, (1, 2, 0))
    img_uint8 = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    
    blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    blurred_np = np.array(blurred_pil).astype(np.float32) / 255.0
    blurred_chw = np.transpose(blurred_np, (2, 0, 1))
    
    return blurred_chw

def apply_random_resize_pad(image_np):
    c, h, w = image_np.shape
    tensor_img = torch.from_numpy(image_np)
    
    import random
    scale_factor = random.uniform(0.95, 0.98)
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    resized_img = F.resize(tensor_img, [new_h, new_w], antialias=True)
    
    pad_h = h - new_h
    pad_w = w - new_w
    pad_top = random.randint(0, pad_h)
    pad_bottom = pad_h - pad_top
    pad_left = random.randint(0, pad_w)
    pad_right = pad_w - pad_left
    
    padded_img = F.pad(resized_img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0)
    
    return padded_img.numpy()

def apply_bit_depth_reduction(image_np, target_bits):
    quant_levels = (2 ** target_bits) - 1
    squeezed_img = np.round(image_np * quant_levels) / quant_levels
    return squeezed_img.astype(np.float32)

def save_debug_image(img_np, save_path):
    img_hwc = np.transpose(img_np, (1, 2, 0))
    img_uint8 = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(save_path)

# ==========================================
# 2. Main Evaluation Pipeline
# ==========================================

class RobustnessEvaluator:
    def __init__(self, config, output_dir, run_id):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f">>> Initializing Model Wrapper: {config['model']['name']}...")
        self.wrapper = get_model_wrapper(config)
        self.model_name = config['model']['name']
        
        self.defenses = {
            'JPEG': [95, 93, 90, 75],
            'Blur': [0.3, 0.5, 1.0],
            'BitDepth': [7, 6, 5],
            'RndPad': [1] 
        }
        
        # 1. Structured Data Logging (CSV)
        self.csv_path = os.path.join(self.output_dir, f"robustness_{self.model_name}_{run_id}.csv")
        with open(self.csv_path, 'w') as f:
            # [NEW] Distinguish between Clean Origin and Adv Origin
            header = ["Image", "Type", "Original_Len", "Original_Time"]
            for q in self.defenses['JPEG']: header.extend([f"JPEG_{q}_Len", f"JPEG_{q}_Time"])
            for r in self.defenses['Blur']: header.extend([f"Blur_{r}_Len", f"Blur_{r}_Time"])
            for b in self.defenses['BitDepth']: header.extend([f"Bit_{b}_Len", f"Bit_{b}_Time"])
            header.extend(["RndPad_Len", "RndPad_Time"])
            f.write(",".join(header) + "\n")

        # 2. Text Content Logging (TXT)
        self.text_log_path = os.path.join(self.output_dir, f"response_trace_{self.model_name}_{run_id}.log")
        with open(self.text_log_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Robustness Response Trace | Model: {self.model_name} ===\n\n")

    def _safe_decode(self, seq):
        """Compatible decoder for various models"""
        if hasattr(self.wrapper, 'safe_decode'):
            return self.wrapper.safe_decode(seq)
        else:
            try:
                decoder = getattr(self.wrapper, 'processor', getattr(self.wrapper, 'tokenizer', None))
                if decoder:
                    seq_list = seq.cpu().tolist() if torch.is_tensor(seq) else seq
                    return decoder.decode(seq_list, skip_special_tokens=True)
                return "[No Decoder]"
            except Exception as e:
                return f"[Decode Error: {e}]"

    def _evaluate_single(self, img_np):
        """Single image inference, returns generated token length, elapsed time, and text content"""
        batch_img = img_np[np.newaxis, ...]
        
        start_time = time.time()
        inputs = self.wrapper._process_inputs(batch_img)
        with torch.no_grad():
            out_ids = self.wrapper.generate(inputs)
            
        seq = out_ids[0]
        length = self.wrapper.get_token_length(seq)
        text = self._safe_decode(seq)
        elapsed = time.time() - start_time
        
        return length, elapsed, text

    def _log_text(self, img_name, img_type, defense_name, length, text):
        """Write to text log"""
        entry = f"[{img_name} | {img_type} | {defense_name}] Len: {length}\n{text[:300]}...\n{'-'*60}\n"
        with open(self.text_log_path, 'a', encoding='utf-8') as f:
            f.write(entry)

    def process_image(self, img_path, img_type, save_debug=False):
        """
        Core processing logic, supports both Clean and Adv types.
        img_type: 'Clean' or 'Adv'
        """
        img_name = os.path.basename(img_path)
        row_data = [img_name, img_type]
        
        try:
            base_img = load_image_for_art(img_path)
            
            # 0. Undefended Baseline
            ud_len, ud_time, ud_text = self._evaluate_single(base_img)
            row_data.extend([ud_len, f"{ud_time:.2f}"])
            self._log_text(img_name, img_type, "Undefended", ud_len, ud_text)
            
            # 1. JPEG Compression
            for q in self.defenses['JPEG']:
                defended_img = apply_jpeg_compression(base_img.copy(), quality=q)
                d_len, d_time, d_text = self._evaluate_single(defended_img)
                row_data.extend([d_len, f"{d_time:.2f}"])
                self._log_text(img_name, img_type, f"JPEG_{q}", d_len, d_text)
                if save_debug: save_debug_image(defended_img, os.path.join(self.output_dir, f"sample_{img_type}_jpeg_{q}.png"))
            
            # 2. Gaussian Blur
            for r in self.defenses['Blur']:
                defended_img = apply_gaussian_blur(base_img.copy(), radius=r)
                d_len, d_time, d_text = self._evaluate_single(defended_img)
                row_data.extend([d_len, f"{d_time:.2f}"])
                self._log_text(img_name, img_type, f"Blur_{r}", d_len, d_text)
                if save_debug: save_debug_image(defended_img, os.path.join(self.output_dir, f"sample_{img_type}_blur_{r}.png"))
                    
            # 3. Bit-Depth Reduction
            for b in self.defenses['BitDepth']:
                defended_img = apply_bit_depth_reduction(base_img.copy(), target_bits=b)
                d_len, d_time, d_text = self._evaluate_single(defended_img)
                row_data.extend([d_len, f"{d_time:.2f}"])
                self._log_text(img_name, img_type, f"Bit_{b}", d_len, d_text)
                if save_debug: save_debug_image(defended_img, os.path.join(self.output_dir, f"sample_{img_type}_bit_{b}.png"))
            
            # 4. Random Resize & Pad
            defended_img = apply_random_resize_pad(base_img.copy())
            d_len, d_time, d_text = self._evaluate_single(defended_img)
            row_data.extend([d_len, f"{d_time:.2f}"])
            self._log_text(img_name, img_type, "RndPad", d_len, d_text)
            if save_debug: save_debug_image(defended_img, os.path.join(self.output_dir, f"sample_{img_type}_rndpad.png"))

            # Write to CSV
            with open(self.csv_path, 'a') as f:
                f.write(",".join(map(str, row_data)) + "\n")
                
        except Exception as e:
            print(f"\n[ERROR] Failed processing {img_name} ({img_type}): {e}")
            import traceback
            traceback.print_exc()

    def run(self, clean_paths, adv_paths):
        print(f"\n=== Starting Defense Evaluation ===")
        print(f"Model: {self.model_name}")
        print(f"Clean Images: {len(clean_paths)}, Adv Images: {len(adv_paths)}")
        
        # Build a dictionary of Adv images for quick pair lookup
        # Assuming clean is "test_001.jpg", adv is "hsja_result_test_001.png" or in a corresponding folder
        # For general purpose, we match by filename (without extension)
        adv_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in adv_paths}
        
        for idx, c_path in enumerate(tqdm(clean_paths)):
            c_name = os.path.splitext(os.path.basename(c_path))[0]
            
            # 1. Evaluate Clean Image
            self.process_image(c_path, img_type="Clean", save_debug=(idx==0))
            
            # 2. Try to find and evaluate corresponding Adv Image
            # In actual file structures, you may need to adjust the matching logic here
            # For example, after hsja_phase2, the adv image might be named 'hsja_result.png' located in a folder named after the clean image
            # Here is a general lookup method:
            a_path = None
            
            # Try to match directly by filename
            if c_name in adv_dict:
                a_path = adv_dict[c_name]
            else:
                # Try fuzzy matching (e.g., if c_name is '001', a_path contains '001')
                for k, v in adv_dict.items():
                    if c_name in k or c_name in v:
                        a_path = v
                        break
            
            if a_path:
                self.process_image(a_path, img_type="Adv", save_debug=(idx==0))
            else:
                print(f"\n[WARNING] Could not find paired Adv image for {c_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/configs.yaml")
    
    # [NEW] Dual path input
    parser.add_argument("--clean_dir", type=str, required=True, help="Directory containing original clean images")
    parser.add_argument("--adv_dir", type=str, required=True, help="Directory containing generated adversarial images")
    
    parser.add_argument("--output_root", type=str, default="data/results/defense_evaluation")
    parser.add_argument("--model_name", type=str, required=True, help="e.g., llava-v1.5-7b")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    args = parser.parse_args()
    
    # 0. Fix global seed
    set_global_seed(args.seed)
    
    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    config['model']['name'] = args.model_name
    if args.model_path:
        config['model']['path'] = args.model_path
        
    # 2. Recursively get image lists
    clean_files = []
    for root, _, files in os.walk(args.clean_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                clean_files.append(os.path.join(root, f))
    clean_files.sort()
                
    adv_files = []
    for root, _, files in os.walk(args.adv_dir):
        for f in files:
            # Filter based on your naming convention, e.g., 'hsja_result.png'
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                adv_files.append(os.path.join(root, f))
    adv_files.sort()
    
    if not clean_files: raise ValueError(f"No clean images found in {args.clean_dir}")
    if not adv_files: raise ValueError(f"No adv images found in {args.adv_dir}")
        
    # 3. Set output directory with timestamp
    import datetime
    run_id = datetime.datetime.now().strftime("%m%d_%H%M")
    output_dir = os.path.join(args.output_root, f"{args.model_name}_{run_id}")
    
    # 4. Execute evaluation
    evaluator = RobustnessEvaluator(config, output_dir, run_id)
    evaluator.run(clean_files, adv_files)
    
    print(f"\n[DONE] Defense Evaluation finished. Results saved to {output_dir}")

if __name__ == "__main__":
    main()