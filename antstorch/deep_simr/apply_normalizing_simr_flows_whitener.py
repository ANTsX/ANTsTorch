from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch

def _ensure_list(x):
    return x if isinstance(x, list) else [x]

def _standardize_with_stats(x_np: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = x_np.copy()
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.broadcast_to(mean, x.shape)[bad]
    std_safe = np.where((np.isfinite(std)) & (std > 0), std, 1.0)
    mean_safe = np.where(np.isfinite(mean), mean, 0.0)
    z = (x - mean_safe) / std_safe
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z

def _destandardize_with_stats(z_np: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std_safe = np.where((np.isfinite(std)) & (std > 0), std, 1.0)
    mean_safe = np.where(np.isfinite(mean), mean, 0.0)
    x = z_np * std_safe + mean_safe
    return x

def _to_df(arr: np.ndarray, like_df: pd.DataFrame, col_prefix: str) -> pd.DataFrame:
    cols = [f"{col_prefix}{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=like_df.index, columns=cols)

def _extract_whitened_from_z(model, z: torch.Tensor) -> torch.Tensor:
    q0 = model.q0
    if hasattr(q0, "W") and hasattr(q0, "loc"):
        return torch.matmul(z - q0.loc, q0.W.T).double()
    loc = getattr(q0, "loc", None)
    scale = getattr(q0, "scale", None)
    if loc is not None and scale is not None:
        return ((z - loc) / (scale + 1e-12)).double()
    return z.double()

def _z_from_whitened(model, eps: torch.Tensor) -> torch.Tensor:
    q0 = model.q0
    if hasattr(q0, "W") and hasattr(q0, "loc"):
        return torch.matmul(eps, q0.W) + q0.loc
    loc = getattr(q0, "loc", None)
    scale = getattr(q0, "scale", None)
    if (loc is not None) and (scale is not None):
        return eps * scale + loc
    return eps

def apply_normalizing_simr_flows_whitener(
    trainer_output: Dict[str, Any] | List[torch.nn.Module],
    data: pd.DataFrame | List[pd.DataFrame],
    direction: str = "forward",              # {"forward", "inverse"}
    use_training_standardization: bool = True,
    custom_standardizers: Dict[int, Dict[str, np.ndarray]] | List[Dict[str, np.ndarray]] | None = None,
    output_space: str = "z",                 # {"z", "whitened"} for direction="forward"
    input_space: str = "z",                  # {"z", "whitened"} for direction="inverse"
    batch_size: int = 4096,
    device: str = "cpu",
    verbose: bool = True,
) -> pd.DataFrame | List[pd.DataFrame]:
    """
    Apply trained normalizing-flow whitener model(s) to new data.
    """
    if isinstance(trainer_output, dict):
        models = trainer_output.get("models", None)
        if models is None:
            raise ValueError("trainer_output dict does not contain 'models'.")
        embedded = trainer_output.get("standardizers", None)
        if (embedded is not None) and custom_standardizers is None:
            custom_standardizers = embedded
    else:
        models = trainer_output

    models = _ensure_list(models)
    data_list = _ensure_list(data)
    if len(models) != len(data_list):
        raise ValueError(f"Number of models ({len(models)}) must match number of views ({len(data_list)}).")

    stdz_map: Dict[int, Dict[str, np.ndarray]] = {}
    if custom_standardizers is not None:
        if isinstance(custom_standardizers, dict) and all(isinstance(k, int) for k in custom_standardizers.keys()):
            stdz_map = {k: {"mean": np.asarray(v["mean"]), "std": np.asarray(v["std"])} for k, v in custom_standardizers.items()}
        else:
            stdz_map = {i: {"mean": np.asarray(d["mean"]), "std": np.asarray(d["std"])} for i, d in enumerate(_ensure_list(custom_standardizers))}

    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    for m in models:
        m.to(device_t).double()
        m.eval()

    outputs: List[pd.DataFrame] = []
    for v_idx, (df, model) in enumerate(zip(data_list, models)):
        X_np = df.to_numpy().astype(np.float64, copy=False)

        used_stats = None
        if direction == "forward" and use_training_standardization:
            if (v_idx not in stdz_map) or ("mean" not in stdz_map[v_idx]) or ("std" not in stdz_map[v_idx]):
                raise ValueError("No mean/std provided for view {v_idx}.")
            mean = stdz_map[v_idx]["mean"]
            std  = stdz_map[v_idx]["std"]
            X_proc = _standardize_with_stats(X_np, mean, std)
            used_stats = (mean, std)
        else:
            X_proc = X_np

        Z_out = []
        with torch.inference_mode():
            for start in range(0, X_proc.shape[0], batch_size):
                stop = min(start + batch_size, X_proc.shape[0])
                xb = torch.from_numpy(X_proc[start:stop]).to(device_t).double()

                if direction == "forward":
                    z = model.forward(xb)
                    if output_space == "whitened":
                        z = _extract_whitened_from_z(model, z)
                    Z_out.append(z.detach().cpu().numpy())

                elif direction == "inverse":
                    q0 = model.q0

                    if input_space == "whitened":
                        # Expect N x L, where L = latent_dim = q0.W.shape[0] for GaussianPCA
                        if hasattr(q0, "W"):
                            L, D = q0.W.shape  # latent_dim, data_dim
                            if xb.shape[1] != L:
                                raise ValueError(
                                    f"apply(): input_space='whitened' expects latent_dim={L}, "
                                    f"but got {xb.shape[1]}. Hint: if you passed raw z, set input_space='z'."
                                )
                        # Map whitened -> z
                        z = _z_from_whitened(model, xb)

                    elif input_space == "z":
                        # Expect N x D, where D = data_dim
                        expected_D = None
                        if hasattr(q0, "loc") and q0.loc is not None:
                            expected_D = int(q0.loc.shape[-1])
                        elif hasattr(q0, "W"):
                            expected_D = int(q0.W.shape[1])  # (L, D)
                        if expected_D is not None and xb.shape[1] != expected_D:
                            raise ValueError(
                                f"apply(): input_space='z' expects data_dim={expected_D}, "
                                f"but got {xb.shape[1]}. Hint: if you passed whitened latents, set input_space='whitened'."
                            )
                        z = xb

                    else:
                        raise ValueError("input_space must be 'z' or 'whitened'")

                    xrec = model.inverse(z)

                    Z_out.append(xrec.detach().cpu().numpy())

        Y = np.vstack(Z_out)

        if (direction == "inverse") and use_training_standardization:
            if (v_idx not in stdz_map) or ("mean" not in stdz_map[v_idx]) or ("std" not in stdz_map[v_idx]):
                raise ValueError("No mean/std provided for view {v_idx} to de-standardize outputs.")
            Y = _destandardize_with_stats(Y, stdz_map[v_idx]["mean"], stdz_map[v_idx]["std"])

        col_prefix = "whitened_" if (direction == "forward" and output_space == "whitened") else ("z_" if direction == "forward" else "xrec_")
        outputs.append(_to_df(Y, df, col_prefix))

    return outputs[0] if isinstance(data, pd.DataFrame) else outputs
