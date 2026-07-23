"""Homomorphic Encryption helpers using Paillier (phe library).

This module provides a lightweight HE layer for federated averaging
of model parameters. It is *additively homomorphic*, which is enough
for summing model updates in FedAvg.
"""

from typing import Dict, List, Tuple
from phe import paillier
import numpy as np
import torch

SCALE = 1e5  # scale factor to convert floats -> ints

def init_context() -> Tuple[paillier.PaillierPublicKey, paillier.PaillierPrivateKey]:
    """Generate a Paillier keypair (public, private)."""
    public_key, private_key = paillier.generate_paillier_keypair()
    return public_key, private_key

def _tensor_to_int_array(t: torch.Tensor, scale: float = SCALE) -> np.ndarray:
    arr = t.detach().cpu().numpy().astype(np.float64)
    arr = np.round(arr * scale).astype(np.int64)
    return arr

def _int_array_to_tensor(arr: np.ndarray, shape, device, scale: float = SCALE) -> torch.Tensor:
    """
    Convert decrypted integer array back to a tensor with given shape.
    Handles both normal tensors and scalars (shape == []).
    """
    arr = arr.astype(np.float32) / scale
    t = torch.tensor(arr, dtype=torch.float32, device=device)

    # shape is usually a torch.Size; convert to tuple
    shape = tuple(shape)

    # Scalar parameter (e.g. BatchNorm num_batches_tracked)
    if len(shape) == 0:
        return t.squeeze()

    # Normal tensor parameter
    return t.view(*shape)


def encrypt_tensor(public_key: paillier.PaillierPublicKey, t: torch.Tensor):
    """Encrypt a tensor element-wise and return encrypted list + original shape."""
    arr = _tensor_to_int_array(t)
    flat = arr.ravel()
    enc_list = [public_key.encrypt(int(v)) for v in flat]
    return enc_list, t.shape

def decrypt_tensor(private_key: paillier.PaillierPrivateKey, enc_list, shape, device) -> torch.Tensor:
    """Decrypt a list of encrypted numbers back into a tensor."""
    dec = [private_key.decrypt(c) for c in enc_list]
    arr = np.array(dec, dtype=np.float32)
    return _int_array_to_tensor(arr, shape, device)

def add_encrypted_lists(lists_of_enc):
    """Element-wise add encrypted numbers from multiple clients.

    lists_of_enc: List[List[EncryptedNumber]]
    """
    if not lists_of_enc:
        return []

    # Start from first client's list
    agg = lists_of_enc[0]
    for other in lists_of_enc[1:]:
        agg = [a + b for a, b in zip(agg, other)]
    return agg

def fedavg_he(
    public_key: paillier.PaillierPublicKey,
    private_key: paillier.PaillierPrivateKey,
    state_dicts: List[Dict[str, torch.Tensor]],
    device,
) -> Dict[str, torch.Tensor]:
    """Perform FedAvg on model parameters using Paillier HE.

    Steps:
    - For each parameter tensor:
      - Encrypt all client tensors element-wise.
      - Add ciphertexts homomorphically.
      - Decrypt aggregated sum.
      - Divide by number of clients to get the averaged tensor.
    """
    agg_state: Dict[str, torch.Tensor] = {}
    num_clients = len(state_dicts)
    keys = state_dicts[0].keys()

    for key in keys:
        enc_lists = []
        shape = state_dicts[0][key].shape

        for sd in state_dicts:
            t = sd[key].to(device)
            enc_list, _ = encrypt_tensor(public_key, t)
            enc_lists.append(enc_list)

        agg_enc = add_encrypted_lists(enc_lists)
        agg_tensor = decrypt_tensor(private_key, agg_enc, shape, device)
        agg_tensor = agg_tensor / num_clients
        agg_state[key] = agg_tensor.cpu()

    return agg_state
