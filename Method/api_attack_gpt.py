import os
import time
import copy
import random
import asyncio
import base64
import io
import json
import numpy as np
from PIL import Image
from openai import AsyncOpenAI
from tqdm import tqdm
import tiktoken 

# ==========================================
# 1. API & Attack Configuration
# ==========================================
# Use environment variables or placeholders.
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") 
MODEL = "gpt-5" 

# Testing Parameters
IMAGE_PATH = "test_dataset/test_001.jpg"  
OUTPUT_DIR = f"api_attack_results_{MODEL.replace('.', '_')}"
PROMPT = "Describe this image in detail."

# Attack Hyperparameters 
POP_SIZE = 32              # Population size 
GENERATIONS = 10           # Iteration generations 
MAX_TARGET_TOKENS = 4096    # Max tokens allowed by API 
CONCURRENCY_LIMIT = 1      # Max concurrent API requests 

# ==========================================
# 2. Utility Functions & Base Classes
# ==========================================
# Initialize tiktoken for fallback token estimation if API doesn't return usage
enc = tiktoken.get_encoding("cl100k_base")

def calc_l2(img_a, img_b):
    return np.linalg.norm(img_a.flatten() - img_b.flatten())

def load_image_as_numpy(path):
    img = Image.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    return img_np.transpose(2, 0, 1)

def save_numpy_image(img_np, save_path):
    img_hwc = img_np.transpose(1, 2, 0)
    img_uint8 = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(save_path)

def numpy_to_base64(img_np):
    img_hwc = img_np.transpose(1, 2, 0)
    img_uint8 = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

class EliteTexturePool:
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.pool = []

    def add(self, gene_list):
        num_to_save = max(1, int(len(gene_list) * 0.4))
        selected_patches = random.sample(gene_list, num_to_save)
        for patch in selected_patches:
            texture_copy = patch['texture'].copy()
            if len(self.pool) >= self.capacity:
                self.pool.pop(0) 
            self.pool.append(texture_copy)

    def sample(self):
        if not self.pool: return None
        return random.choice(self.pool).copy()

    def is_empty(self): return len(self.pool) == 0

# ==========================================
# 3. Core Attack Class
# ==========================================
class UniversalPatchEOGenAttackAPI:
    def __init__(self, img_id="default"):
        self.img_id = img_id
        
        # Parameter Configuration
        self.pop_size = POP_SIZE
        self.generations = GENERATIONS
        self.top_k = max(2, int(POP_SIZE * 0.25))
        self.penalty_beta = 2.0
        
        self.num_patches = 20
        self.patch_scale_ratio = 0.1
        self.patch_alpha = 1.0 
        self.internal_tex_res = 32
        
        self.max_len = MAX_TARGET_TOKENS
        self.elite_pool = EliteTexturePool(capacity=100)
        
        # Statistics
        self.query_count = 0
        self.start_time = 0
        
        self.async_client = None
        self.semaphore = None
        
        # Output Path Configuration
        self.task_dir = os.path.join(OUTPUT_DIR, self.img_id)
        os.makedirs(self.task_dir, exist_ok=True)
        self.detail_log_file = os.path.join(self.task_dir, "attack_detail.log")
        self.progress_log_file = os.path.join(self.task_dir, "attack_progress.csv")
        
        with open(self.detail_log_file, 'w') as f:
            f.write(f"API RUPE Attack on {self.img_id} | Model: {MODEL}\n")
            f.write(f"Target Max Tokens: {self.max_len} | Pop: {self.pop_size}\n{'='*60}\n")
            
        if not os.path.exists(self.progress_log_file):
            with open(self.progress_log_file, 'w') as f:
                f.write("Gen,MaxLen,AvgLen,PoolSize,BestScore,L2Dist,IterTime,TotalTime,Queries\n")

    def _initialize_population(self):
        population = []
        for i in range(self.pop_size):
            individual = []
            use_center_bias = (i < self.pop_size // 2)
            for _ in range(self.num_patches):
                if use_center_bias:
                    rx = np.clip(random.gauss(0.5, 0.2), 0.0, 1.0)
                    ry = np.clip(random.gauss(0.5, 0.2), 0.0, 1.0)
                else:
                    rx = random.random()
                    ry = random.random()
                gene = {
                    'rel_x': rx, 'rel_y': ry,
                    'texture': np.random.rand(3, self.internal_tex_res, self.internal_tex_res).astype(np.float32)
                }
                individual.append(gene)
            population.append(individual)
        return population

    def _rasterize(self, base_image, population):
        batch_size = len(population)
        c, h, w = base_image.shape
        min_dim = min(h, w)
        target_patch_size = int(min_dim * self.patch_scale_ratio)
        if target_patch_size < self.internal_tex_res: target_patch_size = self.internal_tex_res
        scale_factor = target_patch_size // self.internal_tex_res
        if scale_factor < 1: scale_factor = 1
        images = np.tile(base_image[np.newaxis, ...], (batch_size, 1, 1, 1))
        
        for i, individual in enumerate(population):
            for patch in individual:
                rel_x, rel_y, raw_tex = patch['rel_x'], patch['rel_y'], patch['texture']
                upsampled_tex = raw_tex.repeat(scale_factor, axis=1).repeat(scale_factor, axis=2)
                tex_c, tex_h, tex_w = upsampled_tex.shape
                half_w, half_h = tex_w // 2, tex_h // 2
                center_x, center_y = int(rel_x * w), int(rel_y * h)
                x1, y1 = max(0, center_x - half_w), max(0, center_y - half_h)
                x2, y2 = min(w, center_x + half_w), min(h, center_y + half_h)
                tex_x_start = max(0, -(center_x - half_w))
                tex_y_start = max(0, -(center_y - half_h))
                draw_w, draw_h = x2 - x1, y2 - y1
                
                if draw_w > 0 and draw_h > 0:
                    tex_part = upsampled_tex[:, tex_y_start:tex_y_start+draw_h, tex_x_start:tex_x_start+draw_w]
                    bg_part = images[i, :, y1:y2, x1:x2]
                    blended = bg_part * (1 - self.patch_alpha) + tex_part * self.patch_alpha
                    images[i, :, y1:y2, x1:x2] = blended
        return images

    def _crossover(self, parents):
        children = []
        target_count = self.pop_size - len(parents)
        for _ in range(target_count):
            p1, p2 = random.sample(parents, 2)
            child = []
            for i in range(self.num_patches):
                child.append(copy.deepcopy(p1[i] if random.random() < 0.5 else p2[i]))
            children.append(child)
        return children

    def _mutation(self, children, gen_progress):
        mutated_children = []
        pos_sigma = 0.05 * (1.0 - 0.8 * gen_progress)
        tex_sigma = 0.1 * (1.0 - 0.8 * gen_progress)
        
        for child in children:
            m_child = []
            for patch in child:
                new_patch = {'rel_x': patch['rel_x'], 'rel_y': patch['rel_y'], 'texture': patch['texture'].copy()}
                if not self.elite_pool.is_empty() and random.random() < 0.15:
                    new_patch['texture'] = self.elite_pool.sample()
                    m_child.append(new_patch)
                    continue
                if random.random() < 0.05:
                    new_patch['rel_x'], new_patch['rel_y'] = random.random(), random.random()
                elif random.random() < 0.35:
                    new_patch['rel_x'] = np.clip(new_patch['rel_x'] + random.gauss(0, pos_sigma), 0.0, 1.0)
                    new_patch['rel_y'] = np.clip(new_patch['rel_y'] + random.gauss(0, pos_sigma), 0.0, 1.0)
                if random.random() < 0.3:
                    noise = np.random.normal(0, tex_sigma, new_patch['texture'].shape)
                    new_patch['texture'] = np.clip(new_patch['texture'] + noise, 0.0, 1.0)
                m_child.append(new_patch)
            mutated_children.append(m_child)
        return mutated_children

    async def _async_call_api(self, img_np, index):
        async with self.semaphore:
            # numpy_to_base64 internally generates JPEG format, so MIME type image/jpeg is safe
            b64_image = numpy_to_base64(img_np)
            retries = 2
            
            for attempt in range(retries):
                try:
                    response = await self.async_client.chat.completions.create(
                        model=MODEL,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                            ]
                        }],
                        temperature=0,   # Control generation consistency
                        seed=42,         # Fix seed to constrain GPT randomness
                        max_tokens=self.max_len,
                        timeout=120
                    )
                    
                    text = response.choices[0].message.content
                    
                    # Core Improvement: Robust Token Calculation Fallback Mechanism
                    # If proxy returns standard usage, use it; otherwise estimate locally
                    if hasattr(response, "usage") and response.usage and getattr(response.usage, "completion_tokens", None):
                        token_len = response.usage.completion_tokens
                    else:
                        token_len = len(enc.encode(text))
                        
                    return index, token_len, text
                    
                except Exception as e:
                    print(f"  [API Warn] Individual {index} fail: {str(e)[:50]}... Retry {attempt+1}/{retries}")
                    await asyncio.sleep(2)
            
            return index, 5, "[API Error or Safety Blocked]"

    async def _evaluate_population_async(self, rendered_imgs):
        tasks = [self._async_call_api(rendered_imgs[i], i) for i in range(len(rendered_imgs))]
        results = await asyncio.gather(*tasks)
        
        results.sort(key=lambda x: x[0])
        all_lengths = [r[1] for r in results]
        all_texts = [r[2] for r in results]
        
        all_scores = []
        for valid_len in all_lengths:
            if valid_len < self.max_len:
                penalty = self.penalty_beta * (self.max_len - valid_len)
                score = valid_len - penalty
            else:
                score = valid_len * 1.5
            all_scores.append(score)
            
        return np.array(all_scores), np.array(all_lengths), all_texts

    def log_result(self, gen, max_len, avg_len, best_score, best_text, l2_dist, iter_time, total_time):
        detail_entry = (
            f"[Gen {gen}/{self.generations}] MaxLen: {max_len} | Avg: {avg_len:.1f} | "
            f"L2: {l2_dist:.2f} | IterTime: {iter_time:.2f}s | TotalTime: {total_time:.2f}s | "
            f"Q: {self.query_count} | Pool: {len(self.elite_pool.pool)}\n"
            f"--- Best Response Preview ---\n{best_text}\n{'-'*80}\n"
        )
        with open(self.detail_log_file, 'a', encoding='utf-8') as f: f.write(detail_entry)
        csv_line = f"{gen},{max_len},{avg_len:.2f},{len(self.elite_pool.pool)},{best_score:.2f},{l2_dist:.4f},{iter_time:.4f},{total_time:.4f},{self.query_count}\n"
        with open(self.progress_log_file, 'a') as f: f.write(csv_line)

    async def run(self, base_image):
        print(f"[Start] Attack on {self.img_id} (Model: {MODEL})")
        self.start_time = time.time()
        
        self.async_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        
        print("  Evaluating Clean Image (Gen 0)...")
        self.query_count += 1
        
        clean_scores, clean_lengths, clean_texts = await self._evaluate_population_async([base_image])
        clean_len = clean_lengths[0]
        clean_score = clean_scores[0]
        clean_text = clean_texts[0]
        clean_elapsed = time.time() - self.start_time
        
        self.log_result(0, clean_len, float(clean_len), clean_score, clean_text, 0.0, clean_elapsed, clean_elapsed)
        print(f"  [Gen 0] Base Token Length: {clean_len}")
        
        population = self._initialize_population()
        best_overall_len = 0
        best_overall_gene = None
        
        for gen in range(1, self.generations + 1):
            iter_start_time = time.time()
            gen_progress = gen / self.generations
            
            rendered_imgs = self._rasterize(base_image, population)
            self.query_count += self.pop_size
            
            scores, lengths, texts = await self._evaluate_population_async(rendered_imgs)
            
            max_len = np.max(lengths)
            best_idx = np.argmax(scores)
            current_best_score = scores[best_idx]
            current_best_gene = population[best_idx]
            best_text = texts[best_idx]
            
            current_best_img = rendered_imgs[best_idx]
            current_l2 = calc_l2(base_image, current_best_img)
            
            iter_time = time.time() - iter_start_time
            total_time = time.time() - self.start_time
            
            print(f"  [Gen {gen}/{self.generations}] Max Tokens: {max_len} | API Qs: {self.query_count} | Iter Time: {iter_time:.1f}s")
            self.log_result(gen, max_len, np.mean(lengths), current_best_score, best_text, current_l2, iter_time, total_time)
            
            if lengths[best_idx] > best_overall_len:
                best_overall_len = lengths[best_idx]
                best_overall_gene = copy.deepcopy(current_best_gene)
                save_numpy_image(current_best_img, os.path.join(self.task_dir, f"rupe_best_gen{gen}.png"))
                
                # Early stopping mechanism
                if best_overall_len >= self.max_len - 10:
                    print("  [Target] Context Saturated! Early stopping.")
                    break
            
            if lengths[best_idx] >= best_overall_len * 0.90:
                self.elite_pool.add(current_best_gene)
            
            top_indices = np.argsort(scores)[-self.top_k:]
            parents = [copy.deepcopy(population[i]) for i in top_indices]
            children = self._crossover(parents)
            children = self._mutation(children, gen_progress)
            population = parents + children

        print(f"[Success] Attack completed. Max generated token length: {best_overall_len}")
        return self._rasterize(base_image, [best_overall_gene])[0]

# ==========================================
# 4. Main Execution Entry
# ==========================================
async def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"[Warning] Cannot find {IMAGE_PATH}, generating a dummy test image.")
        os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(IMAGE_PATH)

    print("Loading image...")
    base_image = load_image_as_numpy(IMAGE_PATH)
    img_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    attacker = UniversalPatchEOGenAttackAPI(img_id=img_name)
    best_adv_img = await attacker.run(base_image)
    
    final_path = os.path.join(attacker.task_dir, "final_adversarial.png")
    save_numpy_image(best_adv_img, final_path)
    print(f"Results saved to: {attacker.task_dir}")

if __name__ == "__main__":
    asyncio.run(main())
