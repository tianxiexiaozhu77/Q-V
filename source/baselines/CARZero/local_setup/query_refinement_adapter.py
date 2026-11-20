"""Frozen CARZero support-based query refinement with its two scoring directions.

This is a model-specific transfer, not an assertion that CARZero has RadZero's
cosine-aligned feature space. Original-query native DQN attention pools visual
memory into class-conditioned prototypes. Their support positive-minus-negative
direction adjusts the global text query, in both native DQN uses. Local text
tokens and the learned DQN/MLP remain frozen and unchanged.
"""
from pathlib import Path
import os
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "local_setup"))
from query_refinement_common import center_pixels, fit_directions, refine_query


def _serial(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _serial(v) for k, v in value.items()}
    return value


class CARZeroQueryRefinement:
    def __init__(self, native):
        self.native = native
        self.model = native.model
        self.device = native.device
        self.eta = float(os.environ.get("QUERY_REFINEMENT_ETA", ".3"))
        if not np.isfinite(self.eta) or self.eta < 0:
            raise ValueError("QUERY_REFINEMENT_ETA must be finite and nonnegative")
        self.fits = {}
        self.metadata = {
            "model": "CARZero", "eta": self.eta, "gradient_training": False,
            "support_feature": "original-query native image-to-text DQN attention, arithmetic mean across decoder layers, weighted pool of original image global+local memory",
            "space": "native 768-dimensional fusion memory/query latent coordinates; experimental transfer, not proven cosine alignment",
            "query_injection": "same refined global disease text query used as i2t query and first t2i memory token",
            "retained": "original local text tokens, global/local visual tokens, two DQN directions and native arithmetic-mean logits",
            "query_norm": "preserve each original global text query norm",
            "view": "center80 crops native preprocessed pixels and resizes to native input dimensions",
            "supported_tasks": ["classification"],
        }

    def set_dataset(self, entry):
        if hasattr(self.native, "set_dataset"):
            self.native.set_dataset(entry)

    @torch.no_grad()
    def _images(self, image_paths, view="identity"):
        return center_pixels(self.model.process_img(image_paths, self.device), view)

    @staticmethod
    def _memory(local, global_):
        patches = local.view(local.size(0), local.size(1), -1).permute(0, 2, 1)
        return torch.cat([global_.unsqueeze(1), patches], 1)

    @torch.no_grad()
    def _support_features(self, image_paths, class_names, query, batch_size):
        features = []
        for start in range(0, len(image_paths), batch_size):
            images = self._images(image_paths[start:start + batch_size])
            local, global_ = self.model.image_encoder_forward(images)
            memory = self._memory(local, global_)
            _, native_attention = self.model.fusion_module(memory, query, return_atten=True)
            if not isinstance(native_attention, (list, tuple)) or not native_attention:
                raise RuntimeError("Expected native CARZero decoder-layer attention list")
            attention = torch.stack(list(native_attention), 0).mean(0)
            expected_shape = (len(images), len(class_names), memory.shape[1])
            if tuple(attention.shape) != expected_shape:
                raise RuntimeError(f"Unexpected native attention shape: {attention.shape}, expected {expected_shape}")
            if not torch.isfinite(attention).all():
                raise RuntimeError("Nonfinite native support attention")
            features.append(torch.bmm(attention, memory).float().cpu())
        return torch.cat(features)

    @torch.no_grad()
    def fit(self, image_paths, labels, class_names, batch_size=8, task="classification",
            max_iter=1, query_tol=1e-4, patience=1, _state=None):
        if task != "classification":
            raise ValueError("CARZero has no verified native spatial-map adapter")
        if len(image_paths) != len(labels) or not image_paths or batch_size < 1:
            raise ValueError("Invalid support image/label count or batch size")
        if int(max_iter) < 1 or int(max_iter) > 32 or int(patience) < 1 or int(patience) > int(max_iter):
            raise ValueError("Invalid CARZero fixed-point iteration settings")
        if not np.isfinite(query_tol) or query_tol < 0:
            raise ValueError("query_tol must be finite and nonnegative")
        started = __import__("time").perf_counter()
        _, original_query, _ = self.native._text(class_names)
        if _state is None:
            query = original_query.clone()
            initial = final_fit = None
            history, stable = [], 0
            previous_query = previous_corrected = None
            baseline_margin = None
            trajectory = [query.detach().clone()]
        else:
            # A support gate probes q^(1)->q^(2) and, only when selected,
            # continues from q^(2) to the configured full cap.  Keeping this
            # state avoids recomputing the probe and makes the cost claim
            # auditable.
            query = _state["query"].detach().clone().to(self.device)
            initial = _state.get("initial")
            final_fit = _state.get("final_fit")
            history = list(_state.get("history", []))
            stable = 0
            previous_query = _state.get("previous_query")
            previous_corrected = _state.get("previous_corrected")
            baseline_margin = _state.get("baseline_margin")
            trajectory = [x.detach().clone() for x in _state.get("trajectory", [query])]
        start_iteration = len(history)
        # A selected support gate has already computed the native q^(2)
        # support representation in the probe.  Reuse that final direction
        # for the first continuation update; otherwise the gate would pay a
        # duplicate q^(2) native pass and could not claim pre-gate savings.
        reuse_state_fit = _state is not None and _state.get("final_fit") is not None
        for iteration in range(start_iteration, int(max_iter)):
            if reuse_state_fit:
                final_fit = _state["final_fit"]
                support_margin = torch.as_tensor(
                    history[-1].get("support_margin_per_class", []), dtype=torch.float32)
                reuse_state_fit = False
            else:
                features = self._support_features(image_paths, class_names, query, batch_size)
                final_fit = fit_directions(features, torch.as_tensor(labels))
                labels_t = torch.as_tensor(labels, dtype=torch.float32)
                normalized_features = torch.nn.functional.normalize(features, dim=-1)
                normalized_query = torch.nn.functional.normalize(query.detach().cpu(), dim=-1)
                projections = (normalized_features * normalized_query.unsqueeze(0)).sum(-1)
                positives = labels_t.sum(0).clamp_min(1.0)
                negatives = (1.0 - labels_t).sum(0).clamp_min(1.0)
                support_margin = ((projections * labels_t).sum(0) / positives
                                  - (projections * (1.0 - labels_t)).sum(0) / negatives)
            if initial is None:
                initial = final_fit
            # A native support-only margin proxy in the representation that
            # actually drives CARZero's query transport.  This is evaluated
            # before the update (M_S(q^t)); no test image or test label enters
            # the gate signal.
            labels_t = torch.as_tensor(labels, dtype=torch.float32)
            if baseline_margin is None:
                baseline_margin = support_margin.detach().clone()
            corrected = refine_query(original_query, final_fit["directions"].to(self.device),
                                     eta=self.eta, preserve_norm=True)
            delta = torch.linalg.vector_norm(corrected - query, dim=-1)
            if previous_query is None:
                local_lipschitz = None
            else:
                input_step = torch.linalg.vector_norm(query - previous_query, dim=-1)
                output_step = torch.linalg.vector_norm(corrected - previous_corrected, dim=-1)
                local_lipschitz = float((output_step / input_step.clamp_min(1e-8)).mean().cpu())
            history.append({"iteration": iteration, "query_delta_max": float(delta.max().cpu()),
                            "query_delta_mean": float(delta.mean().cpu()),
                            "support_margin_mean": float(support_margin.mean().cpu()),
                            "support_margin_min": float(support_margin.min().cpu()),
                            "support_margin_per_class": support_margin.tolist(),
                            "support_utility": float((support_margin - baseline_margin).mean().cpu()),
                            "local_lipschitz_mean": local_lipschitz,
                            "native_support_probe": True})
            previous_query, previous_corrected = query.detach().clone(), corrected.detach().clone()
            query = corrected
            trajectory.append(query.detach().clone())
            stable = stable + 1 if float(delta.max()) <= float(query_tol) else 0
            if stable >= int(patience):
                break
        # Strict d(q_hat) pass for the representation-inconsistency diagnostic.
        final_features = self._support_features(image_paths, class_names, query, batch_size)
        final_fit = fit_directions(final_features, torch.as_tensor(labels))
        gap = torch.linalg.vector_norm(initial["directions"] - final_fit["directions"], dim=-1)
        cosine = (initial["directions"] * final_fit["directions"]).sum(-1)
        labels_t = torch.as_tensor(labels, dtype=torch.float32)
        nf = torch.nn.functional.normalize(final_features, dim=-1)
        nq = torch.nn.functional.normalize(query.detach().cpu(), dim=-1)
        proj = (nf * nq.unsqueeze(0)).sum(-1)
        pos = labels_t.sum(0).clamp_min(1.0)
        neg = (1.0 - labels_t).sum(0).clamp_min(1.0)
        final_margin = (proj * labels_t).sum(0) / pos - (proj * (1.0 - labels_t)).sum(0) / neg
        q1_margin = history[1].get("support_margin_mean") if len(history) > 1 else None
        q2_margin = float(final_margin.mean()) if len(history) > 1 else None
        gate_delta = None if q1_margin is None or q2_margin is None else q2_margin - float(q1_margin)
        gate_kappa = history[1].get("local_lipschitz_mean") if len(history) > 1 else None
        state = {"query": query.detach().clone(), "initial": initial,
                 "final_fit": final_fit, "history": history,
                 "previous_query": previous_query, "previous_corrected": previous_corrected,
                 "baseline_margin": baseline_margin, "trajectory": trajectory}
        self._fit_state = getattr(self, "_fit_state", {})
        self._fit_state[tuple(class_names)] = state
        self.fits[tuple(class_names)] = corrected
        return _serial({
            **self.metadata, "task": task, "class_names": list(class_names),
            "support_count": len(image_paths), "fit": final_fit,
            "original_queries": original_query, "refined_queries": query,
            "query_trajectory": torch.stack(trajectory),
            "max_iter": int(max_iter), "iterations": len(history),
            "stop_reason": "query_stability" if stable >= int(patience) else "max_iter",
            "history": history, "direction_gap_l2": gap, "direction_gap_l2_mean": float(gap.mean()),
            "direction_gap_l2_median": float(gap.median()), "direction_gap_l2_max": float(gap.max()),
            "direction_gap_cos_mean": float(cosine.mean()), "direction_gap_cos_min": float(cosine.min()),
            "wall_clock_seconds": __import__("time").perf_counter() - started,
            "fit_elapsed_seconds": __import__("time").perf_counter() - started,
            "native_reaggregation_passes": len(history) + 1,
            "estimated_flops": float((len(history) + 1) * len(image_paths) * len(class_names) * original_query.shape[-1] * 2),
            "support_gate_signals": {
                "support_margin_q1": q1_margin,
                "support_margin_q2": q2_margin,
                "support_margin_delta_q2_minus_q1": gate_delta,
                "local_map_ratio": gate_kappa,
                "test_labels_used_for_gate": False,
                "support_probe_passes": 2 if len(history) > 1 else 1,
            },
        })

    def continue_fit(self, image_paths, labels, class_names, batch_size=8,
                     max_iter=3, query_tol=1e-4, patience=1):
        """Continue a support-gate probe without recomputing q^(0)->q^(2)."""
        state = getattr(self, "_fit_state", {}).get(tuple(class_names))
        if state is None:
            raise RuntimeError("fit must run before continue_fit")
        return self.fit(image_paths, labels, class_names, batch_size=batch_size,
                        max_iter=max_iter, query_tol=query_tol, patience=patience,
                        _state=state)

    @torch.no_grad()
    def predict(self, image_paths, class_names, refined=False, view="identity"):
        if not refined and view == "identity":
            return self.native.predict(image_paths, class_names)
        if not image_paths:
            return np.empty((0, len(class_names)), np.float32)
        q_local, q_global, _ = self.native._text(class_names)
        if refined:
            key = tuple(class_names)
            if key not in self.fits:
                raise RuntimeError("fit must be called with this class order first")
            q_global = self.fits[key]
        images = self._images(image_paths, view)
        i_local, i_global = self.model.image_encoder_forward(images)
        return self._score(i_local, i_global, q_local, q_global)

    def _score(self, i_local, i_global, q_local, q_global):
        q_memory = self._memory(q_local, q_global)
        i_memory = self._memory(i_local, i_global)
        outputs = []
        # Match the native adapter's per-image scoring, including its batch
        # structure and score scaling in both directions.
        for i in range(len(i_global)):
            i2t, _ = self.model.fusion_module(i_memory[i:i + 1], q_global, return_atten=True)
            t2i, _ = self.model.fusion_module(q_memory, i_global[i:i + 1], return_atten=True)
            outputs.append((i2t.squeeze(-1) + t2i.squeeze(-1).transpose(1, 0)) / 2)
        return torch.cat(outputs, 0).float().cpu().numpy()

    @torch.no_grad()
    def predict_pair(self, image_paths, class_names, view="identity"):
        if not image_paths:
            return {k: np.empty((0, len(class_names)), np.float32) for k in ("baseline", "refined")}
        q_local, q_global, _ = self.native._text(class_names)
        key = tuple(class_names)
        if key not in self.fits:
            raise RuntimeError("fit must be called with this class order first")
        images = self._images(image_paths, view)
        i_local, i_global = self.model.image_encoder_forward(images)
        return {name: self._score(i_local, i_global, q_local, query)
                for name, query in (("baseline", q_global), ("refined", self.fits[key]))}

    def grounding(self, image_paths, class_names, refined=False, view="identity"):
        raise NotImplementedError("CARZero native spatial-map implementation was not verified; no invented heatmap")


def create_query_adapter(native):
    return CARZeroQueryRefinement(native)
