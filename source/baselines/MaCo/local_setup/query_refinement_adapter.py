"""Frozen MaCo query refinement; original adapter and checkpoints are untouched.

The support prototype lives after MaCo's trained image projection, the same
768-dimensional space used by its raw image/text dot products. Only the positive
query is changed; its norm and all four negative-prompt embeddings are retained.
Center crops are applied to native preprocessed pixels, not to feature maps.
"""
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "local_setup"))
from query_refinement_common import center_pixels, fit_directions, refine_query


def _serial(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _serial(v) for k, v in value.items()}
    return value


class MaCoQueryRefinement:
    def __init__(self, native):
        self.native = native
        self.model = native.model
        self.device = native.device
        self.eta = 0.3
        self.fits = {}
        self.spatial_text_cache = {}
        self.metadata = {
            "model": "MaCo", "eta": self.eta, "gradient_training": False,
            "query_injection": "positive text embedding after trained BERT projection; negative queries fixed",
            "support_feature": "classification: projected mean patch embedding; spatial: native learned-position-weighted projected patches summed",
            "space": "native shared 768-dimensional raw-dot-product image/text space",
            "query_norm": "preserve each original positive query norm",
            "view": "center80 crops native preprocessed pixels and resizes to native input dimensions",
            "spatial": "native weighted raw patch dots, Gaussian 1.5 and nearest interpolation retained; caller aligns center-view maps",
            "supported_tasks": list(native.metadata["supported_tasks"]),
        }

    def set_dataset(self, entry):
        if hasattr(self.native, "set_dataset"):
            self.native.set_dataset(entry)

    @torch.no_grad()
    def _queries(self, class_names, task):
        if task == "classification":
            return self.native._text(class_names)[0]
        key = tuple(class_names)
        if key not in self.spatial_text_cache:
            embeddings = []
            for name in class_names:
                query = f'There is {name.replace("_", " ")}. '
                query = "".join(t.strip() + ". " for t in query.split(".")
                                if len(t.strip() + ". ") > 5).lower()
                tokens = [x.to(self.device) for x in self.model.bert_encoder.encode(query)]
                embeddings.append(self.model.bert_encoder(*tokens).logits.squeeze(0))
            self.spatial_text_cache[key] = torch.stack(embeddings)
        return self.spatial_text_cache[key]

    def _images(self, paths, task, view="identity"):
        spatial_transform = transforms.Compose([
            transforms.Grayscale(3), transforms.ToTensor(),
            transforms.Normalize([.4978], [.2449]),
        ])
        images, sizes = [], []
        for path in paths:
            with Image.open(path) as image:
                sizes.append((image.height, image.width))
                if task == "classification":
                    images.append(self.native.transform(image.convert("RGB")))
                else:
                    images.append(spatial_transform(image.resize(
                        (224, 224), resample=Image.Resampling.BICUBIC)))
        return center_pixels(torch.stack(images).to(self.device), view), sizes

    @torch.no_grad()
    def _features(self, images, task):
        patches = self.model.forward_img_encoder_nomask(images)[:, 1:, :]
        if task == "classification":
            return self.model.img_mlp(patches.mean(1))
        patches = self.model.img_mlp(patches)
        return patches * (self.native.pos_weight_img / .02).softmax(-1).view(1, 196, 1)

    @torch.no_grad()
    def fit(self, image_paths, labels, class_names, batch_size=8, task="classification"):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(image_paths) != len(labels):
            raise ValueError("Support image and label counts differ")
        features = []
        for start in range(0, len(image_paths), batch_size):
            imgs, _ = self._images(image_paths[start:start + batch_size], task)
            feature = self._features(imgs, task)
            if task != "classification":
                feature = feature.sum(1)
            features.append(feature.float().cpu())
        if not features:
            raise ValueError("Support set must not be empty")
        fit = fit_directions(torch.cat(features), torch.as_tensor(labels))
        original = self._queries(class_names, task)
        corrected = refine_query(original, fit["directions"].to(self.device),
                                 eta=self.eta, preserve_norm=True)
        self.fits[(tuple(class_names), task)] = corrected
        return _serial({
            **self.metadata, "task": task, "class_names": list(class_names),
            "support_count": len(image_paths), "fit": fit,
            "original_queries": original, "refined_queries": corrected,
        })

    def _query(self, class_names, task, refined):
        if not refined:
            return self._queries(class_names, task)
        key = (tuple(class_names), task)
        if key not in self.fits:
            raise RuntimeError("fit must be called on this task and class order first")
        return self.fits[key]

    @torch.no_grad()
    def predict(self, image_paths, class_names, refined=False, view="identity"):
        if not refined and view == "identity":
            return self.native.predict(image_paths, class_names)
        if not image_paths:
            return np.empty((0, len(class_names)), np.float32)
        pos = self._query(class_names, "classification", refined)
        neg = self.native._text(class_names)[1]
        images, _ = self._images(image_paths, "classification", view)
        feature = self._features(images, "classification")
        output = torch.stack([feature @ pos.T, feature @ neg.T], -1).softmax(-1)[..., 0]
        return output.float().cpu().numpy()

    @torch.no_grad()
    def predict_pair(self, image_paths, class_names, view="identity"):
        if not image_paths:
            return {k: np.empty((0, len(class_names)), np.float32) for k in ("baseline", "refined")}
        original, neg = self.native._text(class_names)
        corrected = self._query(class_names, "classification", True)
        images, _ = self._images(image_paths, "classification", view)
        feature = self._features(images, "classification")
        negative_logits = feature @ neg.T
        return {key: torch.stack([feature @ query.T, negative_logits], -1).softmax(-1)[..., 0].float().cpu().numpy()
                for key, query in (("baseline", original), ("refined", corrected))}

    @staticmethod
    def _maps(patches, query, sizes):
        from scipy.ndimage import gaussian_filter
        raw = (patches @ query.T).permute(0, 2, 1).reshape(
            patches.shape[0], len(query), 14, 14).float().cpu().numpy()
        smoothed = np.stack([np.stack([
            gaussian_filter(m, sigma=(1.5, 1.5), order=0) for m in row
        ]) for row in raw])
        native_maps = F.interpolate(torch.from_numpy(smoothed), size=(224, 224), mode="nearest")
        return [F.interpolate(native_maps[i:i + 1], size=hw, mode="nearest")[0].numpy()
                for i, hw in enumerate(sizes)]

    def _spatial_task(self, class_names):
        if (tuple(class_names), "grounding") in self.fits:
            return "grounding"
        return "segmentation" if (tuple(class_names), "segmentation") in self.fits else "grounding"

    @torch.no_grad()
    def grounding(self, image_paths, class_names, refined=False, view="identity"):
        if not refined and view == "identity":
            return self.native.grounding(image_paths, class_names)
        if not image_paths:
            return []
        # Grounding and segmentation share identical native maps. The fit task
        # chooses the matching saved support direction without merging supports.
        task = self._spatial_task(class_names)
        query = self._query(class_names, task, refined)
        images, sizes = self._images(image_paths, task, view)
        patches = self._features(images, task)
        return self._maps(patches, query, sizes)

    @torch.no_grad()
    def grounding_pair(self, image_paths, class_names, view="identity"):
        if not image_paths:
            return {"baseline": [], "refined": []}
        task = self._spatial_task(class_names)
        images, sizes = self._images(image_paths, task, view)
        patches = self._features(images, task)
        return {key: self._maps(patches, self._query(class_names, task, refined), sizes)
                for key, refined in (("baseline", False), ("refined", True))}


def create_query_adapter(native):
    return MaCoQueryRefinement(native)
