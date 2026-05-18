"""
Differentiable thermodynamic engine using PyTorch.

This allows optimizing the nearest neighbor parameters using backpropagation.
Computes the exact Partition Function using a batched DP algorithm.
"""

import math
import torch
import torch.nn as nn

from strider.thermo.nn_rna import RNA_NN
from strider.thermo.parameters_rna import (
    TERMINAL_MISMATCH as RNA_TERM_MISMATCH,
    HAIRPIN_SIZE as RNA_HAIRPIN_SIZE,
    INTERIOR_SIZE as RNA_INTERIOR_SIZE,
    BULGE_SIZE as RNA_BULGE_SIZE,
    ASYMMETRY_NINIO as RNA_ASYMMETRY_NINIO,
    DANGLE_3 as RNA_DANGLE_3,
    DANGLE_5 as RNA_DANGLE_5,
    ML_BASE as RNA_ML_BASE,
    ML_INIT as RNA_ML_INIT,
    ML_PAIR as RNA_ML_PAIR
)

INF = 1e9  # Use a large number instead of float('inf') to avoid NaNs in gradients
R = 1.987e-3  # kcal / (mol · K)
T = 310.15 # 37 C
RT = R * T
MIN_HAIRPIN_LOOP = 3

def _can_pair(a: str, b: str, material: str) -> bool:
    if material == "rna":
        wc = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
    else:
        wc = {("A", "T"), ("T", "A"), ("G", "C"), ("C", "G")}
    return (a, b) in wc

def logsumexp_RT(energies, dim=0):
    """
    Computes -RT * ln(sum(exp(-E/RT))) for a list of energy tensors.
    """
    if not energies:
        return None
    if len(energies) == 1:
        return energies[0]
    
    scaled = [-e / RT for e in energies]
    stacked = torch.stack(scaled, dim=dim)
    return -RT * torch.logsumexp(stacked, dim=dim)

class ThermoParameters(nn.Module):
    def __init__(self, material="rna"):
        super().__init__()
        self.material = material
        
        # 1. Stacking Parameters
        dinucs = list(RNA_NN.keys())
        self.dinuc_vocab = dinucs
        self.dinuc_to_idx = {d: i for i, d in enumerate(dinucs)}
        self.stack_dG37 = nn.Parameter(torch.tensor([RNA_NN[d][2] for d in dinucs], dtype=torch.float64))
        self.default_stack = nn.Parameter(torch.tensor(-1.5, dtype=torch.float64))

        # 2. Hairpin Size Penalties
        self.hairpin_sizes = nn.Parameter(torch.tensor(RNA_HAIRPIN_SIZE, dtype=torch.float64))
        
        # 3. Terminal Mismatch
        self.term_mismatch = nn.Parameter(torch.zeros(256, dtype=torch.float64))
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
        for k, v in RNA_TERM_MISMATCH.items():
            idx = base_map[k[0]]*64 + base_map[k[1]]*16 + base_map[k[2]]*4 + base_map[k[3]]
            self.term_mismatch.data[idx] = v
        self.default_term = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        
        # 4. Multiloop penalties
        # New Log-Space Reparameterization:
        # We initialize them as the natural log of their literature values 
        # to ensure they start at the exact same physical baseline.
        self.ml_base_raw = nn.Parameter(torch.log(torch.tensor(max(RNA_ML_BASE, 1e-3), dtype=torch.float64)))
        self.ml_init_raw = nn.Parameter(torch.log(torch.tensor(max(RNA_ML_INIT, 1e-3), dtype=torch.float64)))
        self.ml_pair_raw = nn.Parameter(torch.log(torch.tensor(max(RNA_ML_PAIR, 1e-3), dtype=torch.float64)))

        # 5. Interior Loops & Bulges
        self.interior_sizes = nn.Parameter(torch.tensor(RNA_INTERIOR_SIZE, dtype=torch.float64))
        self.bulge_sizes_raw = nn.Parameter(torch.log(torch.clamp(torch.tensor(RNA_BULGE_SIZE, dtype=torch.float64), min=1e-3)))
        self.asymmetry_ninio = nn.Parameter(torch.tensor(RNA_ASYMMETRY_NINIO, dtype=torch.float64))

        # 6. Neural DP for Interior Loops
        self.mlp_1_1 = nn.Sequential(
            nn.Linear(6 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(torch.float64)
        self.mlp_1_2 = nn.Sequential(
            nn.Linear(7 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(torch.float64)
        self.mlp_2_2 = nn.Sequential(
            nn.Linear(8 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(torch.float64)

        # 7. Dangles
        self.dangle_3 = nn.Parameter(torch.zeros(64, dtype=torch.float64))
        self.dangle_5 = nn.Parameter(torch.zeros(64, dtype=torch.float64))
        for k, v in RNA_DANGLE_3.items():
            if len(k) == 3:
                idx = base_map[k[0]]*16 + base_map[k[1]]*4 + base_map[k[2]]
                self.dangle_3.data[idx] = v
        for k, v in RNA_DANGLE_5.items():
            if len(k) == 3:
                idx = base_map[k[0]]*16 + base_map[k[1]]*4 + base_map[k[2]]
                self.dangle_5.data[idx] = v
                
        # Zero-initialize the final layer of each MLP so they start with 0.0 impact
        for mlp in [self.mlp_1_1, self.mlp_1_2, self.mlp_2_2]:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    def get_hairpin_size_energy(self, loop_len: int) -> torch.Tensor:
        if loop_len < len(self.hairpin_sizes):
            return self.hairpin_sizes[loop_len]
        return self.hairpin_sizes[-1] + 1.75 * RT * math.log(loop_len / float(len(self.hairpin_sizes) - 1))

    def get_interior_size_energy(self, loop_len_tensor: torch.Tensor) -> torch.Tensor:
        idx = loop_len_tensor - 1
        safe_idx = torch.clamp(idx, max=len(self.interior_sizes) - 1)
        base_e = self.interior_sizes[safe_idx]
        extrap_e = self.interior_sizes[-1] + 1.75 * RT * torch.log(loop_len_tensor.float() / float(len(self.interior_sizes)))
        return torch.where(idx < len(self.interior_sizes), base_e, extrap_e)

    def get_bulge_size_energy(self, loop_len_tensor: torch.Tensor) -> torch.Tensor:
        bulge_sizes = torch.exp(self.bulge_sizes_raw)
        idx = loop_len_tensor - 1
        safe_idx = torch.clamp(idx, max=len(bulge_sizes) - 1)
        base_e = bulge_sizes[safe_idx]
        extrap_e = bulge_sizes[-1] + 1.75 * RT * torch.log(loop_len_tensor.float() / float(len(bulge_sizes)))
        return torch.where(idx < len(bulge_sizes), base_e, extrap_e)

class BatchedPartitionFunction(nn.Module):
    def __init__(self, params: ThermoParameters):
        super().__init__()
        self.params = params

    def forward(self, sequences: list[str]) -> torch.Tensor:
        material = self.params.material
        normalized_seqs = []
        for seq in sequences:
            seq = seq.upper()
            seq = seq.replace("U", "T") if material == "dna" else seq.replace("T", "U")
            normalized_seqs.append(seq)
            
        B = len(normalized_seqs)
        max_N = max(len(s) for s in normalized_seqs)
        device = self.params.stack_dG37.device
        
        # Compute strictly positive thermodynamic parameters
        ml_base = torch.exp(self.params.ml_base_raw)
        ml_init = torch.exp(self.params.ml_init_raw)
        ml_pair = torch.exp(self.params.ml_pair_raw)
        
        if max_N == 0:
            return torch.zeros(B, device=device)
            
        padded_seqs = [s + 'X'*(max_N - len(s)) for s in normalized_seqs]
        
        can_pair_mask = torch.zeros(B, max_N, max_N, dtype=torch.bool, device=device)
        dinuc_indices = torch.zeros(B, max_N, dtype=torch.long, device=device)
        valid_dinuc = torch.zeros(B, max_N, dtype=torch.bool, device=device)
        
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'X': 0}
        base_indices = torch.zeros(B, max_N, dtype=torch.long, device=device)
        
        for b, seq in enumerate(padded_seqs):
            for i in range(max_N):
                base_indices[b, i] = base_to_idx.get(seq[i], 0)
                for j in range(max_N):
                    if i < len(sequences[b]) and j < len(sequences[b]):
                        can_pair_mask[b, i, j] = _can_pair(seq[i], seq[j], material)
            
            for i in range(max_N - 1):
                dinuc = seq[i:i+2]
                if dinuc in self.params.dinuc_to_idx:
                    dinuc_indices[b, i] = self.params.dinuc_to_idx[dinuc]
                    valid_dinuc[b, i] = True
        
        # Intermediate Dynamic Programming Array Allocation
        V = torch.full((B, max_N, max_N), INF, dtype=torch.float64, device=device)
        W = torch.zeros((B, max_N, max_N), dtype=torch.float64, device=device) 
        M = torch.full((B, max_N, max_N), INF, dtype=torch.float64, device=device) 
        M1 = torch.full((B, max_N, max_N), INF, dtype=torch.float64, device=device) 
        
        for d in range(1, max_N):
            num_i = max_N - d
            i_idx = torch.arange(num_i, device=device)
            j_idx = i_idx + d
            
            pair_mask = can_pair_mask[:, i_idx, j_idx]
            v_options = []
            
            # 1. Hairpin Loops
            if d > MIN_HAIRPIN_LOOP:
                hp_len = d - 1
                hp_e = self.params.get_hairpin_size_energy(hp_len)
                hp_e_tensor = hp_e.expand(B, num_i)
                
                j_minus_1 = torch.clamp(j_idx - 1, min=0)
                i_plus_1 = torch.clamp(i_idx + 1, max=max_N-1)
                
                idx = base_indices[:, j_minus_1] * 64 + base_indices[:, j_idx] * 16 + base_indices[:, i_idx] * 4 + base_indices[:, i_plus_1]
                term_penalty = self.params.term_mismatch[idx]
                
                v_hp = torch.where(pair_mask, hp_e_tensor + term_penalty, torch.tensor(INF, dtype=torch.float64, device=device))
                v_options.append(v_hp)
                
            # 2. Base Pair Stacking
            if d > MIN_HAIRPIN_LOOP + 1:
                inner_pair_mask = can_pair_mask[:, i_idx+1, j_idx-1]
                valid_stack = pair_mask & inner_pair_mask
                
                stack_params = torch.where(
                    valid_dinuc[:, i_idx],
                    self.params.stack_dG37[dinuc_indices[:, i_idx]],
                    self.params.default_stack
                )
                
                inner_v = V[:, i_idx+1, j_idx-1]
                v_stack = torch.where(valid_stack, stack_params + inner_v, torch.tensor(INF, dtype=torch.float64, device=device))
                v_options.append(v_stack)
                
            # 3. Interior Loops and Bulges
            max_n = min(30, d - 2)
            if max_n >= 1:
                for n in range(1, max_n + 1):
                    d_prime = d - 2 - n
                    if d_prime <= 0:
                        continue
                        
                    nl_tensor = torch.arange(n + 1, device=device)
                    nr_tensor = n - nl_tensor
                    
                    ip_idx = i_idx.unsqueeze(1) + 1 + nl_tensor.unsqueeze(0)
                    jp_idx = j_idx.unsqueeze(1) - 1 - nr_tensor.unsqueeze(0)
                    
                    valid_ip_jp = (ip_idx < max_N) & (jp_idx >= 0) & (ip_idx < jp_idx)
                    ip_safe = torch.clamp(ip_idx, max=max_N-1)
                    jp_safe = torch.clamp(jp_idx, min=0)
                    
                    inner_v = V[:, ip_safe, jp_safe]
                    inner_pair_mask = can_pair_mask[:, ip_safe, jp_safe]
                    valid_interior = valid_ip_jp.unsqueeze(0) & inner_pair_mask & pair_mask.unsqueeze(2)
                    
                    is_bulge = (nl_tensor == 0) | (nr_tensor == 0)
                    n_tensor = torch.tensor(n, device=device)
                    dG_penalty = torch.where(
                        is_bulge,
                        self.params.get_bulge_size_energy(n_tensor),
                        self.params.get_interior_size_energy(n_tensor)
                    )
                    
                    asym = torch.abs(nl_tensor - nr_tensor)
                    ninio_number = torch.clamp(torch.min(nl_tensor, nr_tensor), max=4) - 1
                    ninio_number_safe = torch.clamp(ninio_number, min=0)
                    asym_penalty = torch.min(
                        self.params.asymmetry_ninio[4],
                        asym * self.params.asymmetry_ninio[ninio_number_safe]
                    )
                    dG_penalty = dG_penalty + torch.where(is_bulge, torch.tensor(0.0, dtype=torch.float64, device=device), asym_penalty)
                    dG_penalty_exp = dG_penalty.unsqueeze(0).unsqueeze(0).expand(B, num_i, n + 1)
                    
                    # Neural DP Global Context Fine-Tuning
                    if n == 2:
                        b1, b2 = base_indices[:, i_idx], base_indices[:, j_idx]
                        b3, b4 = base_indices[:, i_idx + 1], base_indices[:, j_idx - 1]
                        b5, b6 = base_indices[:, i_idx + 2], base_indices[:, j_idx - 2]
                        
                        bases = torch.stack([b1, b2, b3, b4, b5, b6], dim=-1)
                        one_hot = torch.nn.functional.one_hot(bases, num_classes=4).to(torch.float64)
                        e_1_1 = self.params.mlp_1_1(one_hot.view(B, num_i, 24)).squeeze(-1)
                        
                        # Ensure the MLP only touches symmetric 1x1 loops, not 0x2 or 2x0 bulges
                        is_true_1_1 = (nl_tensor == 1).unsqueeze(0).unsqueeze(0).expand(B, num_i, n + 1)
                        dG_penalty_exp = dG_penalty_exp + torch.where(is_true_1_1, e_1_1.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float64, device=device))
                        
                    elif n == 3:
                        b1, b2 = base_indices[:, i_idx], base_indices[:, j_idx]
                        b3, b4 = base_indices[:, i_idx + 1], base_indices[:, j_idx - 1]
                        b5, b6 = base_indices[:, j_idx - 2], base_indices[:, i_idx + 2]
                        b7 = base_indices[:, j_idx - 3]
                        
                        bases = torch.stack([b1, b2, b3, b4, b5, b6, b7], dim=-1)
                        one_hot = torch.nn.functional.one_hot(bases, num_classes=4).to(torch.float64)
                        e_1_2 = self.params.mlp_1_2(one_hot.view(B, num_i, 28)).squeeze(-1)
                        
                        # Isolate 1x2 interior loops
                        is_true_1_2 = (nl_tensor == 1).unsqueeze(0).unsqueeze(0).expand(B, num_i, n + 1)
                        dG_penalty_exp = dG_penalty_exp + torch.where(is_true_1_2, e_1_2.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float64, device=device))
                        
                        b1_2, b2_2 = base_indices[:, j_idx], base_indices[:, i_idx]
                        b3_2, b4_2 = base_indices[:, j_idx - 1], base_indices[:, i_idx + 1]
                        b5_2, b6_2 = base_indices[:, i_idx + 2], base_indices[:, j_idx - 2]
                        b7_2 = base_indices[:, i_idx + 3]
                        
                        bases_2 = torch.stack([b1_2, b2_2, b3_2, b4_2, b5_2, b6_2, b7_2], dim=-1)
                        one_hot_2 = torch.nn.functional.one_hot(bases_2, num_classes=4).to(torch.float64)
                        e_2_1 = self.params.mlp_1_2(one_hot_2.view(B, num_i, 28)).squeeze(-1)
                        
                        # Isolate 2x1 interior loops
                        is_true_2_1 = (nl_tensor == 2).unsqueeze(0).unsqueeze(0).expand(B, num_i, n + 1)
                        dG_penalty_exp = dG_penalty_exp + torch.where(is_true_2_1, e_2_1.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float64, device=device))
                        
                    elif n == 4:
                        b1, b2 = base_indices[:, i_idx], base_indices[:, j_idx]
                        b3, b4 = base_indices[:, i_idx + 1], base_indices[:, i_idx + 2]
                        b5, b6 = base_indices[:, j_idx - 1], base_indices[:, j_idx - 2]
                        b7, b8 = base_indices[:, i_idx + 3], base_indices[:, j_idx - 3]
                        
                        bases = torch.stack([b1, b2, b3, b4, b5, b6, b7, b8], dim=-1)
                        one_hot = torch.nn.functional.one_hot(bases, num_classes=4).to(torch.float64)
                        e_2_2 = self.params.mlp_2_2(one_hot.view(B, num_i, 32)).squeeze(-1)
                        
                        # Isolate 2x2 interior loops
                        is_true_2_2 = (nl_tensor == 2).unsqueeze(0).unsqueeze(0).expand(B, num_i, n + 1)
                        dG_penalty_exp = dG_penalty_exp + torch.where(is_true_2_2, e_2_2.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float64, device=device))
                        
                    interior_energies = inner_v + dG_penalty_exp
                    int_scaled = -interior_energies / RT
                    int_scaled = torch.where(valid_interior, int_scaled, torch.tensor(-INF, dtype=torch.float64, device=device))
                    
                    v_int = -RT * torch.logsumexp(int_scaled, dim=-1)
                    v_int = torch.where(torch.isinf(v_int), torch.tensor(INF, dtype=torch.float64, device=device), v_int)
                    v_int = torch.where(pair_mask, v_int, torch.tensor(INF, dtype=torch.float64, device=device))
                    v_options.append(v_int)
                    
            # 4. Multiloop Bifurcation (Autograd-Safe 1D Coordinate Loop)
            L_bif = d - 3
            if L_bif > 0:
                bif_bastes = []
                for m in range(L_bif):
                    k_vals = i_idx + 2 + m
                    v_k = V[:, i_idx + 1, k_vals]
                    w_k = M[:, k_vals + 1, j_idx - 1]
                    bif_bastes.append(v_k + w_k)
                
                bif_energies = torch.stack(bif_bastes, dim=-1)
                ml_penalty = ml_pair
                bif_scaled = -(bif_energies + ml_penalty) / RT
                v_bif = -RT * torch.logsumexp(bif_scaled, dim=-1)
                v_bif = v_bif + ml_init  # Apply ml_init at the final point where multiloop is closed
                v_bif = torch.where(pair_mask, v_bif, torch.tensor(INF, dtype=torch.float64, device=device))
                v_options.append(v_bif)
                
            if v_options:
                V[:, i_idx, j_idx] = logsumexp_RT(v_options, dim=0)
            
            # 5. Populate External Loop (W) and Multiloop (M) Segments
            
            # A. Compute M1 (Exactly one stem, flush right, grows left)
            m1_options = [V[:, i_idx, j_idx] + ml_pair] # Base case: just the stem flush right
            if d >= 1:
                m1_options.append(M1[:, i_idx+1, j_idx] + ml_base) # Add unpaired bases strictly to the left
            M1[:, i_idx, j_idx] = logsumexp_RT(m1_options, dim=0)

            # B. Compute M (One or more stems, grows right)
            m_options = [
                M1[:, i_idx, j_idx],                         # Case 1: It's just a single stem segment
                M[:, i_idx, j_idx-1] + ml_base               # Case 2: Extend trailing unpaired space to the right
            ]

            # Case 3: Multiloop Bifurcation
            for m_w in range(d):
                k_w = i_idx + m_w
                m_options.append(M[:, i_idx, k_w] + M1[:, k_w+1, j_idx])

            M[:, i_idx, j_idx] = logsumexp_RT(m_options, dim=0)

            # C. Compute W (External loop, grows right)
            w_options = [W[:, i_idx, j_idx-1]] # Unpaired on right
            
            for m_w in range(d + 1):
                k_w = i_idx + m_w
                k_minus_1 = k_w - 1
                valid_k_w = k_minus_1 >= i_idx
                k_minus_1_safe = torch.clamp(k_minus_1, min=0)
                
                w_left = W[:, i_idx, k_minus_1_safe]
                w_left = torch.where(valid_k_w, w_left, torch.tensor(0.0, dtype=torch.float64, device=device))
                
                v_right = V[:, k_w, j_idx]
                
                # Keep existing dangle logic for W calculations
                idx_d5 = base_indices[:, k_w] * 16 + base_indices[:, j_idx] * 4 + base_indices[:, k_minus_1_safe]
                dangle_5_penalty = self.params.dangle_5[idx_d5]
                has_d5 = k_w > i_idx
                v_right_d5 = torch.where(has_d5, v_right + dangle_5_penalty, v_right)
                
                w_options.append(w_left + v_right_d5)
                
                if d >= 1:
                    j_minus_1_val = j_idx - 1
                    v_right_d3 = V[:, k_w, j_minus_1_val]
                    idx_d3 = base_indices[:, j_idx] * 16 + base_indices[:, j_minus_1_val] * 4 + base_indices[:, k_w]
                    dangle_3_penalty = self.params.dangle_3[idx_d3]
                    
                    w_k_energies_d3 = w_left + v_right_d3 + dangle_3_penalty
                    valid_k_d3 = k_w <= j_minus_1_val
                    w_k_energies_d3 = torch.where(valid_k_d3, w_k_energies_d3, torch.tensor(INF, dtype=torch.float64, device=device))
                    w_options.append(w_k_energies_d3)
                    
            W[:, i_idx, j_idx] = logsumexp_RT(w_options, dim=0)

        lengths = [len(s) for s in normalized_seqs]
        res = []
        for b, L in enumerate(lengths):
            if L == 0:
                res.append(torch.tensor(0.0, dtype=torch.float64, device=device))
            else:
                res.append(W[b, 0, L - 1])
            
        return torch.stack(res)

DifferentiableMFE = BatchedPartitionFunction