"""Frozen BioViL-T support-query refinement with native classification/maps.

BioViL-T has a shared 128-dimensional global/patch/text space. Support directions
use normalized global embeddings without adding an arbitrary attention mechanism;
the positive text direction also queries the existing native patch heatmaps.
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "local_setup"))
from query_refinement_common import center_pixels, fit_directions, refine_query


class QueryAdapter:
    def __init__(self, native):
        if native.metadata.get("protocol_revision") != "alpharad_binary_v1":
            raise ValueError("BioViL-T refinement requires the frozen binary-protocol adapter")
        self.native = native
        self.device = native.device
        self.eta = 0.3
        self._fit_key = None
        self.metadata = {
            **native.metadata,
            "adaptation": "support positive-minus-negative mean query refinement",
            "eta": self.eta,
            "support_feature": "L2-normalized projected global image embedding, shared 128D space",
            "query_space": "native shared global/patch/text embedding space",
            "negative_queries": "unchanged",
            "gradient_training": False,
            "query_norm": "original positive query norm preserved",
            "grounding_protocol": "native positive-query patch cosine, sigma1.5 smoothing, nearest inverse geometry; center80 inverse/fusion belongs to common evaluator",
            "center_view": "center80 crop on native 448x448 model pixels after original resize512/crop448",
        }

    def _queries(self, class_names):
        phrases = tuple(name.replace("_", " ") for name in class_names)
        if not phrases or any(not name.strip() for name in phrases):
            raise ValueError("Nonempty disease names required")
        if phrases not in self.native._binary_text:
            prompts = [f"There is no {name}." for name in phrases] + [f"There is {name}." for name in phrases]
            self.native._binary_text[phrases] = self.native.text_engine.get_embeddings_from_prompt(prompts, verbose=False)
        return self.native._binary_text[phrases]

    def _images(self, image_paths, view):
        pixels, sizes = self.native._images(image_paths)
        return center_pixels(pixels, view), sizes

    def _check_fit(self, class_names):
        if tuple(class_names) != self._fit_key:
            raise RuntimeError("Fit on the exact ordered support classes before refined inference")

    @torch.inference_mode()
    def fit(self, image_paths, labels, class_names, batch_size=8, task="classification"):
        if task not in ("classification", "grounding", "segmentation"):
            raise ValueError(f"Unknown task: {task}")
        if batch_size < 1 or not image_paths:
            raise ValueError("Positive batch size and nonempty support images required")
        if len(labels) != len(image_paths):
            raise ValueError("Support image/label lengths differ")
        queries = self._queries(class_names)
        spatial_queries = self.native._queries(class_names)
        chunks = []
        for offset in range(0, len(image_paths), batch_size):
            pixels, _ = self._images(image_paths[offset:offset + batch_size], "identity")
            chunks.append(F.normalize(self.native.model(pixels).projected_global_embedding, dim=-1).float().cpu())
        fitted = fit_directions(torch.cat(chunks), labels)
        original = queries[len(class_names):].float().cpu()
        original_spatial = spatial_queries.float().cpu()
        refined = refine_query(original, fitted["directions"], eta=self.eta, preserve_norm=True)
        refined_spatial = refine_query(original_spatial, fitted["directions"], eta=self.eta, preserve_norm=True)
        self._positive = refined.to(self.device)
        self._spatial_positive = refined_spatial.to(self.device)
        self._fit_key = tuple(class_names)
        return {
            **self.metadata, "task": task, "support_count": len(image_paths),
            "class_names": list(class_names),
            "fit": {key: value.tolist() for key, value in fitted.items()},
            "original_positive_queries": original.tolist(),
            "refined_positive_queries": refined.tolist(),
            "original_negative_queries": queries[:len(class_names)].float().cpu().tolist(),
            "original_spatial_queries": original_spatial.tolist(),
            "refined_spatial_queries": refined_spatial.tolist(),
        }

    def _score(self, features, queries, class_count, positive=None):
        all_queries = queries if positive is None else torch.cat((queries[:class_count], positive), dim=0)
        cosine = features @ all_queries.T
        return torch.stack((cosine[:, :class_count], cosine[:, class_count:]), dim=-1).softmax(dim=-1)[..., 1].float().cpu().numpy()

    @torch.inference_mode()
    def predict_pair(self, image_paths, class_names, view="identity"):
        self._check_fit(class_names)
        queries = self._queries(class_names)
        pixels, _ = self._images(image_paths, view)
        features = F.normalize(self.native.model(pixels).projected_global_embedding, dim=-1)
        return {
            "baseline": self._score(features, queries, len(class_names)),
            "refined": self._score(features, queries, len(class_names), self._positive),
        }

    @torch.inference_mode()
    def predict(self, image_paths, class_names, refined=False, view="identity"):
        if not refined and view == "identity":
            return self.native.predict(image_paths, class_names)
        if refined:
            return self.predict_pair(image_paths, class_names, view)["refined"]
        queries = self._queries(class_names)
        pixels, _ = self._images(image_paths, view)
        features = F.normalize(self.native.model(pixels).projected_global_embedding, dim=-1)
        return self._score(features, queries, len(class_names))

    def _maps(self, embeddings, sizes, queries):
        out = []
        for i, (width, height) in enumerate(sizes):
            maps = []
            for query in queries:
                sim = self.native.engine._get_similarity_map_from_embeddings(embeddings[i], query[None, :])
                maps.append(self.native.engine.convert_similarity_to_image_size(
                    sim, width=width, height=height, resize_size=512, crop_size=448,
                    val_img_transform=self.native.transform, interpolation="nearest"))
            out.append(np.stack(maps))
        return out

    @torch.inference_mode()
    def grounding_pair(self, image_paths, class_names, view="identity"):
        self._check_fit(class_names)
        pixels, sizes = self._images(image_paths, view)
        embeddings = self.native.model.get_patchwise_projected_embeddings(pixels, normalize=True)
        return {
            "baseline": self._maps(embeddings, sizes, self.native._queries(class_names)),
            "refined": self._maps(embeddings, sizes, self._spatial_positive),
        }

    @torch.inference_mode()
    def grounding(self, image_paths, class_names, refined=False, view="identity"):
        if not refined and view == "identity":
            return self.native.grounding(image_paths, class_names)
        if refined:
            return self.grounding_pair(image_paths, class_names, view)["refined"]
        pixels, sizes = self._images(image_paths, view)
        embeddings = self.native.model.get_patchwise_projected_embeddings(pixels, normalize=True)
        return self._maps(embeddings, sizes, self.native._queries(class_names))


def create_query_adapter(native):
    return QueryAdapter(native)
