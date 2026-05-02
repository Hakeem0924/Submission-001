import os
import argparse
import numpy as np
import torch
import random
import copy
from PIL import Image
import yaml
import time
from tqdm import tqdm

from models.model_factory import get_model_wrapper
from utils.image_utils import load_image_for_art, save_adversarial_image

def calc_l2(img_a, img_b):
    return np.linalg.norm(img_a.flatten() - img_b.flatten())

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


class UniversalPatchEOGenAttack:
    def __init__(self, wrapper, config, img_id="default"):
        self.wrapper = wrapper
        self.config = config
        self.img_id = img_id
        self.device = wrapper.device
        
        eogen_conf = config.get('eogen', {})
        self.pop_size = eogen_conf.get('pop_size', 32)
        self.generations = eogen_conf.get('generations', 100)
        self.top_k = eogen_conf.get('top_k', 8)
        self.penalty_beta = eogen_conf.get('penalty_beta', 2.0)
        
        self.num_patches = eogen_conf.get('num_attack_patches', 20)
        self.patch_scale_ratio = eogen_conf.get('patch_scale_ratio', 0.12)
        self.patch_alpha = eogen_conf.get('patch_alpha', 0.5) 
        self.internal_tex_res = eogen_conf.get('internal_tex_res', 32)
        
        self.elite_pool = EliteTexturePool(capacity=100)
        self.global_best_score = -float('inf')
        self.max_len = config['model']['generation']['max_new_tokens']
        
        self.query_count = 0
        self.start_time = 0
        
        self.model_name = config['model']['name']
        base_output_dir = config['data']['output_dir']
        self.task_dir = os.path.join(base_output_dir, self.img_id)
        os.makedirs(self.task_dir, exist_ok=True)
        
        self.detail_log_file = os.path.join(self.task_dir, "attack_detail.log")
        self.progress_log_file = os.path.join(self.task_dir, "attack_progress.csv")
        
        with open(self.detail_log_file, 'w') as f:
            f.write(f"RUPE Attack on {self.img_id} | Model: {self.model_name}\n")
            f.write(f"Config: {eogen_conf}\n{'='*60}\n")
            
        if not os.path.exists(self.progress_log_file) or os.path.getsize(self.progress_log_file) == 0:
            with open(self.progress_log_file, 'w') as f:
                # [UPDATE] Add IterTime column
                f.write("Gen,MaxLen,AvgLen,PoolSize,BestScore,L2Dist,IterTime,TotalTime,Queries\n")

    def _initialize_population(self):
        population = []
        for i in range(self.pop_size):
            individual = []
            # Center Bias
            use_center_bias = (i < self.pop_size // 2)
            # use_center_bias = False
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

    def _generate_mask(self, image_shape, gene):
        c, h, w = image_shape
        min_dim = min(h, w)
        mask = np.zeros((c, h, w), dtype=np.float32)
        target_patch_size = int(min_dim * self.patch_scale_ratio)
        if target_patch_size < self.internal_tex_res: target_patch_size = self.internal_tex_res
        half_w = target_patch_size // 2
        half_h = target_patch_size // 2
        for patch in gene:
            rel_x = patch['rel_x']
            rel_y = patch['rel_y']
            center_x = int(rel_x * w)
            center_y = int(rel_y * h)
            x1 = max(0, center_x - half_w)
            y1 = max(0, center_y - half_h)
            x2 = min(w, center_x + half_w)
            y2 = min(h, center_y + half_h)
            if x2 > x1 and y2 > y1:
                mask[:, y1:y2, x1:x2] = 1.0
        return mask

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
                rel_x = patch['rel_x']
                rel_y = patch['rel_y']
                raw_tex = patch['texture']
                upsampled_tex = raw_tex.repeat(scale_factor, axis=1).repeat(scale_factor, axis=2)
                tex_c, tex_h, tex_w = upsampled_tex.shape
                half_w = tex_w // 2
                half_h = tex_h // 2
                center_x = int(rel_x * w)
                center_y = int(rel_y * h)
                x1 = max(0, center_x - half_w)
                y1 = max(0, center_y - half_h)
                x2 = min(w, center_x + half_w)
                y2 = min(h, center_y + half_h)
                tex_x_start = max(0, -(center_x - half_w))
                tex_y_start = max(0, -(center_y - half_h))
                draw_w = x2 - x1
                draw_h = y2 - y1
                if draw_w > 0 and draw_h > 0:
                    tex_part = upsampled_tex[:, tex_y_start:tex_y_start+draw_h, tex_x_start:tex_x_start+draw_w]
                    bg_part = images[i, :, y1:y2, x1:x2]
                    blended = bg_part * (1 - self.patch_alpha) + tex_part * self.patch_alpha
                    images[i, :, y1:y2, x1:x2] = blended
        return images

    def _crossover(self, parents):
        children = []
        target_count = self.pop_size - len(parents)
        if len(parents) < 2:
            for _ in range(target_count):
                parent = random.choice(parents)
                children.append(copy.deepcopy(parent))
            return children
        for _ in range(target_count):
            p1, p2 = random.sample(parents, 2)
            child = []
            for i in range(self.num_patches):
                if random.random() < 0.5:
                    child.append(copy.deepcopy(p1[i]))
                else:
                    child.append(copy.deepcopy(p2[i]))
            children.append(child)
        return children

    def _mutation(self, children, gen_progress):
        mutated_children = []
        pos_sigma = 0.05 * (1.0 - 0.8 * gen_progress)
        tex_sigma = 0.1 * (1.0 - 0.8 * gen_progress)

        for child in children:
            m_child = []
            for patch in child:
                new_patch = {
                    'rel_x': patch['rel_x'],
                    'rel_y': patch['rel_y'],
                    'texture': patch['texture'].copy()
                }
                if not self.elite_pool.is_empty() and random.random() < 0.15:
                    elite_tex = self.elite_pool.sample()
                    if elite_tex is not None:
                        new_patch['texture'] = elite_tex
                        m_child.append(new_patch)
                        continue
                
                # Global Jump
                if random.random() < 0.05:
                    new_patch['rel_x'] = random.random()
                    new_patch['rel_y'] = random.random()
                elif random.random() < 0.35:
                    dx = random.gauss(0, pos_sigma)
                    dy = random.gauss(0, pos_sigma)
                    new_patch['rel_x'] = np.clip(new_patch['rel_x'] + dx, 0.0, 1.0)
                    new_patch['rel_y'] = np.clip(new_patch['rel_y'] + dy, 0.0, 1.0)
                
                if random.random() < 0.3:
                    noise = np.random.normal(0, tex_sigma, new_patch['texture'].shape)
                    new_patch['texture'] = np.clip(new_patch['texture'] + noise, 0.0, 1.0)
                
                m_child.append(new_patch)
            mutated_children.append(m_child)
        return mutated_children

    def _evaluate(self, population, base_image):
        rendered_imgs = self._rasterize(base_image, population)
        all_scores = []
        all_lengths = []
        batch_size = self.config['model']['batch_size']
        current_gen_best_text = ""
        current_gen_max_score = -float('inf')
        num_imgs = len(population)
        self.query_count += num_imgs
        
        for i in range(0, num_imgs, batch_size):
            batch_imgs = rendered_imgs[i : i + batch_size]
            inputs = self.wrapper._process_inputs(batch_imgs)
            with torch.no_grad():
                output_ids = self.wrapper.generate(inputs)
            for seq in output_ids:
                valid_len = self.wrapper.get_token_length(seq)
                if valid_len < self.max_len:
                    penalty = self.penalty_beta * (self.max_len - valid_len)
                    score = valid_len - penalty
                else:
                    score = valid_len * 1.5
                all_scores.append(score)
                all_lengths.append(valid_len)
                if score > current_gen_max_score:
                    current_gen_max_score = score
                    if hasattr(self.wrapper, 'safe_decode'):
                        current_gen_best_text = self.wrapper.safe_decode(seq)
                    else:
                        try:
                            decoder = getattr(self.wrapper, 'processor', getattr(self.wrapper, 'tokenizer', None))
                            if decoder:
                                seq_list = seq.cpu().tolist() if torch.is_tensor(seq) else seq
                                current_gen_best_text = decoder.decode(seq_list, skip_special_tokens=True)
                            else:
                                current_gen_best_text = "[No Decoder]"
                        except Exception as e:
                            current_gen_best_text = f"[Decode Exception: {str(e)}]"
        return np.array(all_scores), np.array(all_lengths), current_gen_best_text

    def log_result(self, gen, max_len, avg_len, best_score, best_text, l2_dist, iter_time, total_time):
        # [UPDATE] Add iter_time and total_time
        detail_entry = (
            f"[Gen {gen}/{self.generations}] MaxLen: {max_len} | Avg: {avg_len:.1f} | "
            f"L2: {l2_dist:.2f} | IterTime: {iter_time:.2f}s | TotalTime: {total_time:.2f}s | "
            f"Q: {self.query_count} | Pool: {len(self.elite_pool.pool)}\n"
            f"--- Best Response Preview ---\n{best_text}\n{'-'*80}\n"
        )
        with open(self.detail_log_file, 'a', encoding='utf-8') as f:
            f.write(detail_entry)
            
        csv_line = f"{gen},{max_len},{avg_len:.2f},{len(self.elite_pool.pool)},{best_score:.2f},{l2_dist:.4f},{iter_time:.4f},{total_time:.4f},{self.query_count}\n"
        with open(self.progress_log_file, 'a') as f:
            f.write(csv_line)

    def run(self, base_image):
        self.start_time = time.time()
        
        # Gen 0
        clean_batch = base_image[np.newaxis, ...]
        inputs = self.wrapper._process_inputs(clean_batch)
        self.query_count += 1
        with torch.no_grad():
            output_ids = self.wrapper.generate(inputs)
        seq0 = output_ids[0]
        clean_len = self.wrapper.get_token_length(seq0)
        
        if clean_len < self.max_len:
            penalty = self.penalty_beta * (self.max_len - clean_len)
            clean_score = clean_len - penalty
        else:
            clean_score = clean_len * 1.5
            
        if hasattr(self.wrapper, 'safe_decode'):
            clean_text = self.wrapper.safe_decode(seq0)
        else:
            try:
                decoder = getattr(self.wrapper, 'processor', getattr(self.wrapper, 'tokenizer', None))
                if decoder:
                    seq_list = seq0.cpu().tolist() if torch.is_tensor(seq0) else seq0
                    clean_text = decoder.decode(seq_list, skip_special_tokens=True)
                else:
                    clean_text = "[No Decoder Found]"
            except Exception as e:
                clean_text = f"[Gen0 Decode Error: {e}]"
        
        clean_elapsed = time.time() - self.start_time
        # Gen 0: IterTime = TotalTime
        self.log_result(0, clean_len, float(clean_len), clean_score, clean_text, 0.0, clean_elapsed, clean_elapsed)
        
        # Evolution
        population = self._initialize_population()
        best_overall_len = 0
        best_overall_gene = None
        self.global_best_len = 0 
        
        for gen in range(1, self.generations + 1):
            iter_start_time = time.time() # [NEW] Record generation start time
            
            gen_progress = gen / self.generations
            scores, lengths, best_text = self._evaluate(population, base_image)
            
            max_len = np.max(lengths)
            best_idx = np.argmax(scores)
            current_best_score = scores[best_idx]
            current_best_len_in_gen = lengths[best_idx]
            current_best_gene = population[best_idx]
            
            # Metrics
            current_best_img = self._rasterize(base_image, [current_best_gene])[0]
            current_l2 = calc_l2(base_image, current_best_img)
            
            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time # Generation duration
            total_time = iter_end_time - self.start_time # Total elapsed time
            
            self.log_result(gen, max_len, np.mean(lengths), current_best_score, best_text, current_l2, iter_time, total_time)
            
            if current_best_len_in_gen > best_overall_len:
                best_overall_len = current_best_len_in_gen
                best_overall_gene = copy.deepcopy(current_best_gene)
                self.global_best_len = best_overall_len
                
                save_adversarial_image(current_best_img, 
                    os.path.join(self.task_dir, f"rupe_best_{gen}.png"))
                
                best_mask = self._generate_mask(base_image.shape, best_overall_gene)
                save_adversarial_image(best_mask, 
                    os.path.join(self.task_dir, f"rupe_mask_{gen}.png"))
                
                if best_overall_len >= self.max_len - 10:
                    with open(self.detail_log_file, 'a') as f: 
                        f.write("Context Saturated.\n")
                    break
            
            if current_best_len_in_gen >= self.global_best_len * 0.90:
                self.elite_pool.add(current_best_gene)
            
            top_indices = np.argsort(scores)[-self.top_k:]
            parents = [copy.deepcopy(population[i]) for i in top_indices]
            children = self._crossover(parents)
            children = self._mutation(children, gen_progress)
            population = parents + children

        if not self.elite_pool.is_empty():
            # Retrieve all texture data from the pool
            pool_textures = self.elite_pool.pool
            
            # Export the latest textures up to the specified limit
            # Since the pool is FIFO, elements at the end are newer
            num_to_export = min(len(pool_textures), self.num_patches)
            exported_textures = pool_textures[-num_to_export:]
            
            # Pack textures into a dictionary for np.savez
            save_dict = {f"tex_{i}": tex for i, tex in enumerate(exported_textures)}
            
            pool_save_path = os.path.join(self.task_dir, "elite_texture_pool.npz")
            np.savez_compressed(pool_save_path, **save_dict)
            print(f"[RUPE] Exported {len(exported_textures)} elite textures to {pool_save_path}")
        else:
            print("[RUPE] Elite pool is empty. Nothing to export.")
            
        return self._rasterize(base_image, [best_overall_gene])[0]