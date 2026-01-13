import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_openfold_on_syspath(openfold_dir: Path) -> None:
    openfold_dir = openfold_dir.resolve()
    if not openfold_dir.exists():
        raise FileNotFoundError(
            f"OpenFold directory not found: {openfold_dir}. "
            f"Set OPENFOLD_DIR or pass openfold_dir."
        )
    if str(openfold_dir) not in sys.path:
        sys.path.insert(0, str(openfold_dir))


def _import_openfold_modules():
    from openfold import config
    from openfold.data import feature_pipeline
    from openfold.data import data_pipeline
    from openfold.data import parsers
    from openfold.model import modified_model
    from openfold.np import protein
    from openfold.utils.import_weights import import_jax_weights_
    from openfold.utils.tensor_utils import tensor_tree_map
    from openfold.utils.script_utils import prep_output

    return (
        config,
        feature_pipeline,
        data_pipeline,
        parsers,
        modified_model,
        protein,
        import_jax_weights_,
        tensor_tree_map,
        prep_output,
    )


def _placeholder_template_feats(num_templates: int, num_res: int) -> Dict[str, np.ndarray]:
    return {
        "template_aatype": np.zeros((num_templates, num_res, 22), dtype=np.int64),
        "template_all_atom_positions": np.zeros((num_templates, num_res, 37, 3), dtype=np.float32),
        "template_all_atom_mask": np.zeros((num_templates, num_res, 37), dtype=np.float32),
        "template_domain_names": np.array([b""], dtype=np.bytes_),
        "template_sum_probs": np.zeros((num_templates, 1), dtype=np.float32),
    }


class OpenfoldMixPredictor:
    """
    Mix predictor built on openfold.model.modified_model.AlphaFold.

    Defaults:
      - OpenFold code: <repo_root>/openfold
      - Params dir:    <repo_root>/openfold/params
    """

    def __init__(
        self,
        device: str = "cuda:0",
        precision=None,
        model_idx: int = 3,
        use_ptm_weight: bool = True,
        openfold_dir: Optional[str] = None,
        params_dir: Optional[str] = None,
    ):
        root = _project_root()

        of_dir = Path(openfold_dir) if openfold_dir else Path(os.environ.get("OPENFOLD_DIR", root / "openfold"))
        _ensure_openfold_on_syspath(of_dir)

        pd_dir = Path(params_dir) if params_dir else Path(os.environ.get("ALPHAFOLD_PARAMS_DIR", of_dir / "params"))
        pd_dir = pd_dir.resolve()
        if not pd_dir.exists():
            raise FileNotFoundError(
                f"Params directory not found: {pd_dir}. "
                f"Set ALPHAFOLD_PARAMS_DIR or pass params_dir."
            )

        (
            config,
            feature_pipeline,
            data_pipeline,
            parsers,
            modified_model,
            protein,
            import_jax_weights_,
            tensor_tree_map,
            prep_output,
        ) = _import_openfold_modules()

        self._feature_pipeline = feature_pipeline
        self._data_pipeline = data_pipeline
        self._parsers = parsers
        self._protein = protein
        self._import_jax_weights = import_jax_weights_
        self._tensor_tree_map = tensor_tree_map
        self._prep_output = prep_output

        preset = f"model_{model_idx}"
        self.base_preset = preset
        if use_ptm_weight:
            preset = preset + "_ptm"
        self.config_preset = preset

        self.cfg = config.model_config(preset)
        net = modified_model.AlphaFold(self.cfg).eval()

        params_file = pd_dir / f"params_{preset}.npz"
        if not params_file.exists():
            raise FileNotFoundError(f"Params file not found: {params_file}")

        self._import_jax_weights(net, str(params_file), version=preset)

        if precision is not None:
            net = net.to(precision)

        self.device = device
        self.openfold_model = net.to(device)

    def _set_cfg(self, num_recycles: int, max_msa_clusters: int) -> None:
        self.cfg.data.predict.max_extra_msa = 1
        self.cfg.data.predict.max_msa_clusters = max_msa_clusters
        self.cfg.data.common.max_recycling_iters = num_recycles

    def to_evoformer(
        self,
        a3m_text: str,
        sequence: str,
        num_recycles: int = 3,
        max_msa_clusters: int = 512,
    ) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict, dict, object]:
        self._set_cfg(num_recycles=num_recycles, max_msa_clusters=max_msa_clusters)

        msas = [self._parsers.parse_a3m(a3m_text)]
        num_res = len(sequence)

        feature_dict = {}
        feature_dict.update(self._data_pipeline.make_sequence_features(sequence, "test", num_res))
        feature_dict.update(self._data_pipeline.make_msa_features(msas))
        feature_dict.update(_placeholder_template_feats(1, num_res))

        feature_processor = self._feature_pipeline.FeaturePipeline(self.cfg.data)
        processed = feature_processor.process_features(feature_dict, mode="predict")
        processed = {k: torch.as_tensor(v, device=self.device) for k, v in processed.items()}

        with torch.no_grad():
            outputs, m, z, s = self.openfold_model.to_evoformer(processed)

        return outputs, m, z, s, processed, feature_dict, feature_processor

    def to_IPA(self, outputs, m, z, s, processed_feature_dict):
        with torch.no_grad():
            pred = self.openfold_model.to_IPA(outputs, m, z, s, processed_feature_dict)
        return pred

    def write_pdb(
        self,
        processed_feature_dict: dict,
        prediction_result: dict,
        feature_dict: dict,
        feature_processor,
        pdb_path: str,
        return_plddt: bool = False,
    ):
        processed_np = self._tensor_tree_map(lambda t: np.array(t[..., -1].detach().cpu()), processed_feature_dict)
        pred_np = self._tensor_tree_map(lambda t: np.array(t.detach().cpu()), prediction_result)

        prot = self._prep_output(
            pred_np,
            processed_np,
            feature_dict,
            feature_processor,
            self.config_preset,
            multimer_ri_gap=200,
            subtract_plddt=False,
        )

        with open(pdb_path, "w") as f:
            f.write(self._protein.to_pdb(prot))

        if return_plddt:
            return pred_np["plddt"]
