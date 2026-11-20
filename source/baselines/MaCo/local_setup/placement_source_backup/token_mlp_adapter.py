"""Added text-guided patch aggregation retaining MaCo's raw-dot binary readout.

Native MaCo classification averages patches. New cosine weights act on native
unnormalized img_mlp-projected patches. Positive and negative prompts have separate
aggregates; author prompt means and final raw-dot scale are unchanged.
"""
from pathlib import Path
import importlib.util

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NATIVE_PATH = ROOT / "local_setup/adapter.py"


class TokenBackend:
    dim = 768
    tokens_per_image = 196

    def __init__(self, device="cuda"):
        spec = importlib.util.spec_from_file_location("_maco_token_native", NATIVE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.native = module.create_adapter(device)
        self.device = self.native.device
        self.class_names = None
        if not isinstance(self.native.model.img_mlp, torch.nn.Linear):
            raise RuntimeError("Native mean/project equivalence requires the released linear img_mlp")
        self.metadata = {
            **self.native.metadata, "aggregation_origin": "added", "dim": self.dim,
            "tokens_per_image": self.tokens_per_image,
            "token_boundary": "all 196 visual patch tokens (no CLS), after frozen native linear img_mlp",
            "aggregation": "separate positive/negative cosine softmax, fixed temperature 1; raw projected values; native unnormalized raw-dot positive-minus-negative readout",
            "attention_temperature": 1.0, "native_score_kind": self.native.metadata["score_kind"],
            "score_transform": "positive-minus-negative raw dot margin; exact underlying native binary logit, avoids probability saturation",
            "native_prediction_transform": "torch.logit(probability, eps=1e-7) is diagnostic only and may clamp saturated MaCo scores",
            "zero_adapter_reference": "aggregation_only", "native_aggregation": "mean patch then frozen linear img_mlp",
            "recommended_cache_batch_size": 32, "recommended_train_batch_size": 32,
            "cache_batch_size": 32, "train_microbatch_size": 32,
            "source_path": str(Path(__file__).resolve()), "native_adapter_path": str(NATIVE_PATH),
            "source_files": [str(NATIVE_PATH), str(ROOT / "ZeroShot/model_MaCo.py")],
            "python_launcher": str(ROOT / "local_setup/python_local.sh"),
        }

    @torch.no_grad()
    def prepare_classes(self, class_names):
        self.class_names = tuple(class_names)
        if not self.class_names or any(not isinstance(x, str) or not x.strip() for x in self.class_names):
            raise ValueError("Nonempty ordered class names required")
        positive, negative = self.native._text(self.class_names)
        self.positive = positive.detach().clone()
        self.negative = negative.detach().clone()
        self.queries = torch.cat((self.positive, self.negative), dim=0)
        self.attention_queries = F.normalize(self.queries, dim=-1)
        self.metadata.update(class_names=list(self.class_names), prompts=self.native.metadata["last_prompts"])

    def _images(self, paths):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(self.native.transform(image.convert("RGB")))
        return torch.stack(images).to(self.device)

    @torch.no_grad()
    def encode_images(self, paths):
        patches = self.native.model.forward_img_encoder_nomask(self._images(paths))[:, 1:, :]
        tokens = self.native.model.img_mlp(patches).float()
        if tuple(tokens.shape[1:]) != (self.tokens_per_image, self.dim):
            raise RuntimeError(f"Unexpected projected token shape: {tuple(tokens.shape)}")
        return tokens.detach()

    def score_tokens(self, tokens, token_adapter=None):
        if token_adapter is not None:
            tokens = token_adapter(tokens)
        weights = torch.einsum("bnd,qd->bqn", F.normalize(tokens, dim=-1), self.attention_queries).softmax(-1)
        contexts = torch.einsum("bqn,bnd->bqd", weights, tokens)
        scores = torch.einsum("bqd,qd->bq", contexts, self.queries)
        count = len(self.class_names)
        return scores[:, :count] - scores[:, count:]

    def native_scores_from_tokens(self, tokens):
        features = tokens.mean(1)
        return features @ self.positive.T - features @ self.negative.T

    @torch.no_grad()
    def native_predictions(self, paths):
        return torch.as_tensor(self.native.predict(paths, self.class_names), device=self.device)

    @torch.no_grad()
    def native_scores(self, paths):
        patches = self.native.model.forward_img_encoder_nomask(self._images(paths))[:, 1:, :]
        features = self.native.model.img_mlp(patches.mean(1))
        return features @ self.positive.T - features @ self.negative.T

    def release_image_encoder(self):
        if self.class_names is None:
            raise RuntimeError("prepare_classes must precede encoder release")
        self.native = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
