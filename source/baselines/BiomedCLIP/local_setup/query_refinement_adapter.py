"""Support-derived positive-query refinement for the frozen BiomedCLIP baseline.

No optimizer or test labels are used. The direction is computed in the model's
normalized global image/text projection space, rather than a newly introduced
attention module. Native binary absence/presence scoring stays unchanged.
"""
from pathlib import Path
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "local_setup"))
from query_refinement_common import center_pixels, fit_directions, refine_query


class QueryAdapter:
    def __init__(self, native):
        if native.metadata.get("protocol_revision") != "alpharad_binary_v1":
            raise ValueError("BiomedCLIP refinement requires the frozen binary-protocol adapter")
        self.native = native
        self.device = native.device
        self.eta = float(os.environ.get("QUERY_REFINEMENT_ETA", "0.3"))
        if not torch.isfinite(torch.tensor(self.eta)) or self.eta < 0:
            raise ValueError("QUERY_REFINEMENT_ETA must be finite and nonnegative")
        self._fit_key = None
        self.metadata = {
            **native.metadata,
            "adaptation": "support positive-minus-negative mean query refinement",
            "eta": self.eta,
            "support_feature": "L2-normalized projected global image embedding",
            "query_space": "native shared normalized image/text embedding space",
            "negative_queries": "unchanged",
            "gradient_training": False,
            "query_norm": "original positive query norm preserved",
            "center_view": "center80 crop after native image preprocessing; resize to native model input",
        }

    def _queries(self, class_names):
        phrases = tuple(name.replace("_", " ") for name in class_names)
        if not phrases or any(not name.strip() for name in phrases):
            raise ValueError("Nonempty disease names required")
        if phrases not in self.native._binary_text:
            prompts = [f"There is no {name}." for name in phrases] + [f"There is {name}." for name in phrases]
            tokens = self.native.tokenizer(prompts).to(self.device)
            self.native._binary_text[phrases] = self.native.model.encode_text(tokens, normalize=True)
        return self.native._binary_text[phrases]

    def _images(self, image_paths, view):
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(self.native.preprocess(image.convert("RGB")))
        return center_pixels(torch.stack(images).to(self.device), view)

    def _check_fit(self, class_names):
        if tuple(class_names) != self._fit_key:
            raise RuntimeError("Fit on the exact ordered support classes before refined inference")

    @torch.inference_mode()
    def fit(self, image_paths, labels, class_names, batch_size=8, task="classification"):
        if task != "classification":
            raise NotImplementedError("No verified native BiomedCLIP grounding implementation")
        if batch_size < 1 or not image_paths:
            raise ValueError("Positive batch size and nonempty support images required")
        if len(labels) != len(image_paths):
            raise ValueError("Support image/label lengths differ")
        queries = self._queries(class_names)
        chunks = []
        for offset in range(0, len(image_paths), batch_size):
            pixels = self._images(image_paths[offset:offset + batch_size], "identity")
            chunks.append(self.native.model.encode_image(pixels, normalize=True).float().cpu())
        fitted = fit_directions(torch.cat(chunks), labels)
        original = queries[len(class_names):].float().cpu()
        refined = refine_query(original, fitted["directions"], eta=self.eta, preserve_norm=True)
        self._positive = refined.to(self.device)
        self._fit_key = tuple(class_names)
        return {
            **self.metadata, "task": task, "support_count": len(image_paths),
            "class_names": list(class_names),
            "fit": {key: value.tolist() for key, value in fitted.items()},
            "original_positive_queries": original.tolist(),
            "refined_positive_queries": refined.tolist(),
            "original_negative_queries": queries[:len(class_names)].float().cpu().tolist(),
        }

    def _score(self, features, queries, class_count, positive=None):
        if positive is None:
            cosine = features @ queries.T
            negative_score, positive_score = cosine[:, :class_count], cosine[:, class_count:]
        else:
            all_queries = torch.cat((queries[:class_count], positive), dim=0)
            cosine = features @ all_queries.T
            negative_score, positive_score = cosine[:, :class_count], cosine[:, class_count:]
        return torch.stack((negative_score, positive_score), dim=-1).softmax(dim=-1)[..., 1].float().cpu().numpy()

    @torch.inference_mode()
    def predict_pair(self, image_paths, class_names, view="identity"):
        self._check_fit(class_names)
        queries = self._queries(class_names)
        features = self.native.model.encode_image(self._images(image_paths, view), normalize=True)
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
        features = self.native.model.encode_image(self._images(image_paths, view), normalize=True)
        return self._score(features, queries, len(class_names))


def create_query_adapter(native):
    return QueryAdapter(native)
