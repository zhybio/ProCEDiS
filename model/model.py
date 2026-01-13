import torch
import torch.nn as nn


class SimilarityPredictor(nn.Module):
    """
    Multi-task predictor:
      - Shared encoder: input_embed: [*, L*21] -> [*, hidden]
      - Head-1 (per-MSA): plddt_head: [*, hidden] -> [*, 1] in [0,1]
      - Head-2 (pair):    rmsd_head:  combine(hA,hB) -> [*, 1]  (regression)

    Notes:
      - forward() expects pair inputs [B, 2, L*21]
      - predict_plddt() expects single inputs [B, L*21]
    """

    def __init__(self, res_num: int, hidden_dim: int = 128):
        super().__init__()
        in_dim = int(res_num) * 21

        # Shared embed
        self.input_embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        # Head-1: per-MSA pLDDT in [0,1]
        self.plddt_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Head-2: pair RMSD regression
        self.rmsd_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def embed(self, emb: torch.Tensor) -> torch.Tensor:
        """
        emb:
          - [B, L*21] or
          - [B, 2, L*21]
        return:
          - [B, hidden] or
          - [B, 2, hidden]
        """
        return self.input_embed(emb)

    @torch.no_grad()
    def predict_plddt(self, emb: torch.Tensor) -> torch.Tensor:
        """
        emb: [B, L*21]
        return: [B] in [0,1]
        """
        h = self.embed(emb)                       # [B, hidden]
        p = self.plddt_head(h).squeeze(-1)        # [B]
        return p

    def forward(self, concat_emb: torch.Tensor):
        """
        concat_emb: [B, 2, L*21]
        return:
          rmsd_pred:  [B]
          plddt_pred: [B, 2]   (two MSAs in the pair)
        """
        h = self.embed(concat_emb)                # [B, 2, hidden]

        # per-element pLDDT
        plddt_pred = self.plddt_head(h).squeeze(-1)  # [B, 2]

        # pair RMSD
        h_A, h_B = h[:, 0, :], h[:, 1, :]
        delta = torch.abs(h_A - h_B)
        combined = torch.cat([delta, h_A + h_B, h_A * h_B], dim=-1)  # [B, 3*hidden]
        rmsd_pred = self.rmsd_head(combined).squeeze(-1)             # [B]

        return rmsd_pred, plddt_pred
