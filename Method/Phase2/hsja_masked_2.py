import numpy as np
import os
import time
from art.attacks.evasion import HopSkipJump
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin

class MaskedHopSkipJump(HopSkipJump):
    """
    Enhanced HSJA with Advanced Theoretical Corrections:
    1. Masked Gradient Estimation (Hard Constraint).
    2. Dimension Correction for Delta.
    3. Trust-Region Dynamic Delta Scaling.
    4. Momentum-based Boundary Sliding.
    """
    def __init__(self, classifier, trace_log_path=None, **kwargs):
        if not isinstance(classifier, (BaseEstimator, ClassifierMixin)):
            class MixedClassifier(classifier.__class__, BaseEstimator, ClassifierMixin):
                pass
            classifier.__class__ = MixedClassifier

        super().__init__(classifier=classifier, **kwargs)
        
        self.trace_log_path = trace_log_path
        self.total_queries = 0
        
        self.current_phase = "INIT"  
        self.current_iter = 0
        self.phase_queries = {
            "INIT": 0, "BINARY_SEARCH": 0, "GRADIENT_EST": 0, "GEOMETRIC_PROG": 0
        }
        
        self.last_valid_x_adv = None 
        
        self.historical_grad = 0
        self.momentum_decay = 0.5 
        
        if self.trace_log_path:
            os.makedirs(os.path.dirname(self.trace_log_path), exist_ok=True)
            with open(self.trace_log_path, 'w') as f:
                f.write("Iter,Phase,L2_Dist,Step_Queries,Total_Queries,Event\n")

    def _log_trace(self, event_note="", l2_dist=0.0):
        if self.trace_log_path:
            step_q = self.phase_queries.get(self.current_phase, 0)
            log_line = (f"{self.current_iter},{self.current_phase},{l2_dist:.4f},"
                        f"{step_q},{self.total_queries},{event_note}\n")
            with open(self.trace_log_path, 'a') as f:
                f.write(log_line)

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        self.total_queries = 0
        self.current_iter = 0
        self.last_valid_x_adv = None
        self.historical_grad = 0 
        for k in self.phase_queries: self.phase_queries[k] = 0
        
        return super().generate(x, y, **kwargs)

    def _perturb(self, x, y, y_p, init_pred, adv_init, mask, clip_min, clip_max):
        self.last_valid_x_adv = adv_init if adv_init is not None else x
        if adv_init is None:
             return x 

        x_adv = self._attack(adv_init, x, y, mask, clip_min, clip_max)
        
        if mask is not None:
            x_adv = x_adv * mask + x * (1 - mask)
            
        return x_adv

    def _adversarial_satisfactory(
        self, samples: np.ndarray, target: int, clip_min: float, clip_max: float
    ) -> np.ndarray:
        query_cost = len(samples)
        self.total_queries += query_cost
        if self.current_phase in self.phase_queries:
            self.phase_queries[self.current_phase] += query_cost
            
        return super()._adversarial_satisfactory(samples, target, clip_min, clip_max)

    def _binary_search(self, *args, **kwargs):
        prev_phase = self.current_phase
        self.current_phase = "BINARY_SEARCH"
        self.phase_queries["BINARY_SEARCH"] = 0 
        result = super()._binary_search(*args, **kwargs)
        self.current_phase = prev_phase
        return result

    def _compute_delta(self, current_sample, original_sample, clip_min, clip_max, mask=None):
        if self.curr_iter == 0:
            base_delta = 0.1 * (clip_max - clip_min)
        else:
            d_original = np.prod(self.estimator.input_shape)
            if self.norm == 2:
                dist = np.linalg.norm(original_sample - current_sample)
                base_delta = np.sqrt(d_original) * self.theta * dist
            else:
                dist = np.max(abs(original_sample - current_sample))
                base_delta = d_original * self.theta * dist

        if mask is not None:
            d_original = np.prod(self.estimator.input_shape)
            d_eff = np.sum(mask) 
            if d_eff > 0:
                correction_factor = np.sqrt(d_original / d_eff) 
                corrected_delta = base_delta * correction_factor
                self._log_trace(f"Delta Corrected: {base_delta:.4f} -> {corrected_delta:.4f} (d_eff={d_eff})")
                return corrected_delta
                
        return base_delta

    def _compute_update(
        self, current_sample, num_eval, delta, target, mask, clip_min, clip_max, original_sample=None
    ):
        self.current_phase = "GRADIENT_EST"
        self.phase_queries["GRADIENT_EST"] = 0
        self.last_valid_x_adv = current_sample
        
        delta_min = 1.0 / 255.0 
        if original_sample is not None:
            dist = np.linalg.norm(original_sample - current_sample) if self.norm == 2 else np.max(abs(original_sample - current_sample))
            delta_max = dist / np.sqrt(self.current_iter + 1) 
        else:
            delta_max = delta * 10.0 
            
        current_delta = np.clip(delta, delta_min, delta_max)
        max_retries = 5
        
        for retry in range(max_retries):
            rnd_noise_shape = [num_eval] + list(self.estimator.input_shape)
            if self.norm == 2:
                rnd_noise = np.random.randn(*rnd_noise_shape).astype(ART_NUMPY_DTYPE)
            else:
                rnd_noise = np.random.uniform(low=-1, high=1, size=rnd_noise_shape).astype(ART_NUMPY_DTYPE)

            if mask is not None:
                rnd_noise = rnd_noise * mask

            rnd_noise = rnd_noise / np.sqrt(np.sum(rnd_noise**2, axis=tuple(range(len(rnd_noise_shape)))[1:], keepdims=True))
            
            eval_samples = np.clip(current_sample + current_delta * rnd_noise, clip_min, clip_max)
            actual_noise = (eval_samples - current_sample) / current_delta

            satisfied = self._adversarial_satisfactory(eval_samples, target, clip_min, clip_max)
            f_val = 2 * satisfied.reshape([num_eval] + [1] * len(self.estimator.input_shape)) - 1.0
            f_val = f_val.astype(ART_NUMPY_DTYPE)
            
            mean_f = np.mean(f_val)
            
            if mean_f == -1.0:
                new_delta = min(current_delta * 3.0, delta_max)
                if new_delta > current_delta and retry < max_retries - 1:
                    self._log_trace(f"Gradient Uniform (-1.0). Expanding Delta to {new_delta:.6f}")
                    current_delta = new_delta
                    continue
            elif mean_f == 1.0:
                new_delta = max(current_delta / 2.0, delta_min)
                if new_delta < current_delta and retry < max_retries - 1:
                    self._log_trace(f"Gradient Uniform (1.0). Shrinking Delta to {new_delta:.6f}")
                    current_delta = new_delta
                    continue
            
            if mean_f == 1.0:
                raw_grad = np.mean(actual_noise, axis=0)
            elif mean_f == -1.0:
                raw_grad = -np.mean(actual_noise, axis=0)
            else:
                f_val -= mean_f
                raw_grad = np.mean(f_val * actual_noise, axis=0)

            fused_grad = (1 - self.momentum_decay) * raw_grad + self.momentum_decay * self.historical_grad
            self.historical_grad = fused_grad 
            
            if self.norm == 2:
                result = fused_grad / (np.linalg.norm(fused_grad) + 1e-12) 
            else:
                result = np.sign(fused_grad)
            
            self._log_trace(f"Gradient Computed (Retry {retry}, mean_f={mean_f:.2f})")
            return result

        return result

    def _attack(self, initial_sample, original_sample, target, mask, clip_min, clip_max):
        current_sample = initial_sample

        for i in range(self.max_iter):
            self.current_iter = i
            
            # 1. Delta
            delta = self._compute_delta(current_sample, original_sample, clip_min, clip_max, mask=mask)

            # 2. Binary Search
            current_sample = self._binary_search(
                current_sample=current_sample,
                original_sample=original_sample,
                norm=self.norm,
                target=target,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            l2_after_bin = np.linalg.norm(original_sample - current_sample)
            self._log_trace("Binary Search Done", l2_dist=l2_after_bin)
            
            # 3. Gradient
            num_eval = min(int(self.init_eval * np.sqrt(self.curr_iter + 1)), self.max_eval)
            update = self._compute_update(
                current_sample, num_eval, delta, target, mask, clip_min, clip_max, original_sample
            )

            # 4. Geometric Progression
            self.current_phase = "GEOMETRIC_PROG"
            self.phase_queries["GEOMETRIC_PROG"] = 0
            
            dist = np.linalg.norm(original_sample - current_sample) if self.norm == 2 else np.max(abs(original_sample - current_sample))
            epsilon = 2.0 * dist / np.sqrt(self.curr_iter + 1)
            success = False
            
            while not success:
                epsilon /= 2.0
                potential_sample = current_sample + epsilon * update
                
                success = self._adversarial_satisfactory(
                    samples=potential_sample[None],
                    target=target,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )[0]

            current_sample = np.clip(potential_sample, clip_min, clip_max)
            self.curr_iter += 1
            
            l2_final = np.linalg.norm(original_sample - current_sample)
            self._log_trace(f"Iter {i} Finished", l2_dist=l2_final)
            
        return current_sample