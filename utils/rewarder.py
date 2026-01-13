import numpy as np
import torch


class Rewarder(object):
    """
    RMSD rewarder with batched Kabsch alignment (correct rotation).

    Key:
      - We build covariance H = A^T B  (A=query centered, B=buffer centered).
      - If H = U S V^T, the optimal rotation mapping A -> B is:
            R = U V^T
        (NOT V U^T for this H definition)

    Public API:
      - calculate(query, buffer) -> torch.Tensor [B] on CPU
      - rmsd_to_buffer_torch(query_xyz, buffer_xyz) -> torch.Tensor [B] on same device as inputs
      - align(query_xyz, target_xyz) -> aligned query_xyz (torch.Tensor [N,3])
      - rmsd(coords1, coords2) -> scalar tensor
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    @staticmethod
    def _to_torch_xyz(x, device: torch.device) -> torch.Tensor:
        """
        Accept np/list/torch, return torch.float32 tensor on device, shape [..., 3].
        """
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        x = np.asarray(x, dtype=np.float32)
        return torch.from_numpy(x).to(device=device, dtype=torch.float32)

    def rmsd_to_buffer_torch(self, query_xyz: torch.Tensor, buffer_xyz: torch.Tensor) -> torch.Tensor:
        """
        query_xyz:  [N, 3]
        buffer_xyz: [B, N, 3]
        return:     [B]
        """
        if query_xyz.ndim != 2 or query_xyz.shape[-1] != 3:
            raise ValueError(f"query_xyz must be [N,3], got {tuple(query_xyz.shape)}")
        if buffer_xyz.ndim != 3 or buffer_xyz.shape[-1] != 3:
            raise ValueError(f"buffer_xyz must be [B,N,3], got {tuple(buffer_xyz.shape)}")
        if buffer_xyz.shape[1] != query_xyz.shape[0]:
            raise ValueError(
                f"Residue count mismatch: query N={query_xyz.shape[0]} vs buffer N={buffer_xyz.shape[1]}"
            )

        device = buffer_xyz.device
        query = query_xyz.to(device=device, dtype=torch.float32)
        buf = buffer_xyz.to(device=device, dtype=torch.float32)

        B, N, _ = buf.shape
        A = query.unsqueeze(0).expand(B, -1, -1)  # [B,N,3]

        # Center
        centroid_A = A.mean(dim=1, keepdim=True)      # [B,1,3]
        centroid_B = buf.mean(dim=1, keepdim=True)    # [B,1,3]
        AA = A - centroid_A                           # [B,N,3]
        BB = buf - centroid_B                         # [B,N,3]

        # Covariance: H = A^T B  -> [B,3,3]
        H = torch.matmul(AA.transpose(1, 2), BB)

        # SVD: H = U S Vh (Vh = V^T for real)
        U, S, Vh = torch.linalg.svd(H)

        # Correct rotation for this H: R = U V^T = U @ Vh
        R = torch.matmul(U, Vh)  # [B,3,3]

        # Fix reflection if det(R) < 0:
        # flip last column of U, then recompute R = U V^T
        detR = torch.det(R)
        mask = detR < 0
        if mask.any():
            U2 = U.clone()
            U2[mask, :, -1] *= -1.0
            R = torch.matmul(U2, Vh)

        # Align: (A - cA) R + cB
        A_aligned = torch.matmul(AA, R) + centroid_B  # [B,N,3]

        diff = A_aligned - buf
        mse = (diff ** 2).sum(dim=-1).mean(dim=-1)    # mean over residues of squared distance
        rmsd = torch.sqrt(mse + self.eps)             # [B]
        return rmsd

    def align(self, protA: torch.Tensor, protB: torch.Tensor) -> torch.Tensor:
        """
        Align protA onto protB (single pair), return aligned protA.
        protA, protB: [N,3] torch tensors
        """
        if protA.ndim != 2 or protA.shape[-1] != 3:
            raise ValueError(f"protA must be [N,3], got {tuple(protA.shape)}")
        if protB.ndim != 2 or protB.shape[-1] != 3:
            raise ValueError(f"protB must be [N,3], got {tuple(protB.shape)}")
        if protA.shape[0] != protB.shape[0]:
            raise ValueError(f"N mismatch: {protA.shape[0]} vs {protB.shape[0]}")

        device = protA.device
        A = protA.to(dtype=torch.float32)
        B = protB.to(device=device, dtype=torch.float32)

        cA = A.mean(dim=0, keepdim=True)   # [1,3]
        cB = B.mean(dim=0, keepdim=True)   # [1,3]
        AA = A - cA
        BB = B - cB

        H = AA.t().matmul(BB)              # [3,3] = A^T B
        U, S, Vh = torch.linalg.svd(H)
        R = U.matmul(Vh)                   # R = U V^T

        if torch.det(R) < 0:
            U[:, -1] *= -1.0
            R = U.matmul(Vh)

        A_aligned = AA.matmul(R) + cB
        return A_aligned

    def rmsd(self, coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
        """
        Single-pair RMSD (torch tensors), returns scalar tensor.
        """
        coords1_aligned = self.align(coords1, coords2)
        diff = coords1_aligned - coords2
        mse = (diff ** 2).sum(dim=-1).mean()  # mean over residues of squared distance
        return torch.sqrt(mse + self.eps)

    def calculate(self, query, buffer):
        """
        Backward-compatible API.

        query:  [N,3] numpy/list/torch
        buffer: list/np/torch of shape [B,N,3] (or [N,3] single)
        return: torch.Tensor [B] on CPU
        """
        if buffer is None:
            return torch.zeros((0,), dtype=torch.float32)

        # Normalize buffer to [B,N,3]
        if isinstance(buffer, torch.Tensor):
            buf_t = buffer
            if buf_t.ndim == 2 and buf_t.shape[-1] == 3:
                buf_t = buf_t.unsqueeze(0)
            if buf_t.ndim != 3 or buf_t.shape[-1] != 3:
                raise ValueError(f"buffer must be [B,N,3], got {tuple(buf_t.shape)}")
            device = buf_t.device
        else:
            buf_np = np.asarray(buffer, dtype=np.float32)
            if buf_np.ndim == 2 and buf_np.shape[-1] == 3:
                buf_np = buf_np[None, ...]
            if buf_np.ndim != 3 or buf_np.shape[-1] != 3:
                raise ValueError(f"buffer must be [B,N,3], got {buf_np.shape}")
            device = torch.device("cpu")
            buf_t = torch.from_numpy(buf_np).to(device=device, dtype=torch.float32)

        q_t = self._to_torch_xyz(query, device=device)
        if q_t.ndim != 2 or q_t.shape[-1] != 3:
            raise ValueError(f"query must be [N,3], got {tuple(q_t.shape)}")

        out = self.rmsd_to_buffer_torch(q_t, buf_t)
        return out.detach().cpu()
