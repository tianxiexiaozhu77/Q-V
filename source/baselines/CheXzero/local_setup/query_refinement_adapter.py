"""Support-query refinement independently for each official CheXzero member.

The ten-model ensemble, native preprocessing and per-member positive-negative
sigmoid scores are preserved. No parameters receive gradient updates.
"""
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "local_setup"))
from query_refinement_common import center_pixels, fit_directions, refine_query


class QueryAdapter:
    def __init__(self, native):
        self.native = native
        self.device = native.device
        self.eta = 0.3
        self._fit_key = None
        self.metadata = {
            **native.metadata,
            "adaptation": "independent support positive-minus-negative query refinement per frozen ensemble member",
            "eta": self.eta,
            "support_feature": "L2-normalized projected global image embedding, separately for each of 10 models",
            "query_space": "each ensemble member's own native shared image/text embedding space",
            "negative_queries": "unchanged for every ensemble member",
            "gradient_training": False,
            "query_norm": "original positive query norm preserved",
            "center_view": "center80 crop after native image preprocessing; resize to native 224 model input",
        }

    def _queries(self, class_names):
        key = tuple(class_names)
        if not key or any(not isinstance(name, str) or not name.strip() for name in key):
            raise ValueError("Nonempty disease names required")
        if key not in self.native._text:
            phrases = ["pleural thickening" if name.lower() == "pleural_thickening" else name for name in key]
            tokens = self.native.tokenize(
                [f"There is {name}." for name in phrases] + [f"There is no {name}." for name in phrases],
                context_length=77).to(self.device)
            self.native._text[key] = [F.normalize(model.encode_text(tokens), dim=-1) for model in self.native.models]
        return self.native._text[key]

    def _images(self, image_paths, view):
        pixels = torch.stack([self.native._image(path) for path in image_paths]).to(self.device)
        return center_pixels(pixels, view)

    def _check_fit(self, class_names):
        if tuple(class_names) != self._fit_key:
            raise RuntimeError("Fit on the exact ordered support classes before refined inference")

    @torch.inference_mode()
    def fit(self, image_paths, labels, class_names, batch_size=8, task="classification"):
        if task != "classification":
            raise NotImplementedError("No verified native CheXzero grounding implementation")
        if batch_size < 1 or not image_paths:
            raise ValueError("Positive batch size and nonempty support images required")
        if len(labels) != len(image_paths):
            raise ValueError("Support image/label lengths differ")
        query_list = self._queries(class_names)
        chunks = [[] for _ in self.native.models]
        for offset in range(0, len(image_paths), batch_size):
            pixels = self._images(image_paths[offset:offset + batch_size], "identity")
            for i, model in enumerate(self.native.models):
                chunks[i].append(F.normalize(model.encode_image(pixels), dim=-1).float().cpu())
        self._positive = []
        members = []
        for index, (features, queries) in enumerate(zip(chunks, query_list)):
            fitted = fit_directions(torch.cat(features), labels)
            original = queries[:len(class_names)].float().cpu()
            refined = refine_query(original, fitted["directions"], eta=self.eta, preserve_norm=True)
            self._positive.append(refined.to(self.device))
            members.append({
                "member_index": index, "checkpoint": str(self.native.checkpoints[index]),
                "fit": {key: value.tolist() for key, value in fitted.items()},
                "original_positive_queries": original.tolist(),
                "refined_positive_queries": refined.tolist(),
                "original_negative_queries": queries[len(class_names):].float().cpu().tolist(),
            })
        self._fit_key = tuple(class_names)
        return {
            **self.metadata, "task": task, "support_count": len(image_paths),
            "class_names": list(class_names), "members": members,
        }

    @torch.inference_mode()
    def predict_pair(self, image_paths, class_names, view="identity"):
        self._check_fit(class_names)
        queries_list = self._queries(class_names)
        pixels = self._images(image_paths, view)
        baseline, refined = [], []
        count = len(class_names)
        for model, queries, positive in zip(self.native.models, queries_list, self._positive):
            features = F.normalize(model.encode_image(pixels), dim=-1)
            original_similarity = features @ queries.T
            refined_similarity = features @ torch.cat((positive, queries[count:]), dim=0).T
            baseline.append(torch.sigmoid(original_similarity[:, :count] - original_similarity[:, count:]))
            refined.append(torch.sigmoid(refined_similarity[:, :count] - refined_similarity[:, count:]))
        return {
            "baseline": torch.stack(baseline).mean(0).float().cpu().numpy(),
            "refined": torch.stack(refined).mean(0).float().cpu().numpy(),
        }

    @torch.inference_mode()
    def predict(self, image_paths, class_names, refined=False, view="identity"):
        if not refined and view == "identity":
            return self.native.predict(image_paths, class_names)
        if refined:
            return self.predict_pair(image_paths, class_names, view)["refined"]
        queries_list = self._queries(class_names)
        pixels = self._images(image_paths, view)
        count = len(class_names)
        predictions = []
        for model, queries in zip(self.native.models, queries_list):
            features = F.normalize(model.encode_image(pixels), dim=-1)
            similarity = features @ queries.T
            predictions.append(torch.sigmoid(similarity[:, :count] - similarity[:, count:]))
        return torch.stack(predictions).mean(0).float().cpu().numpy()


def create_query_adapter(native):
    return QueryAdapter(native)
