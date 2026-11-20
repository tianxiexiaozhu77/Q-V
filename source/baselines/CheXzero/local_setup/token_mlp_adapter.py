"""Added token aggregation preserving all ten native CheXzero ensemble members.

One residual MLP is shared across tokens, classes AND ensemble members. Their
embedding coordinate systems need not be aligned; this is an explicit modeling
choice, not ten independently trained adapters. Zero MLP equals aggregation-only,
not native CLS classification. Frozen CLS projection is extended to patches.
"""
from pathlib import Path
import importlib.util

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NATIVE_PATH = ROOT / "local_setup/adapter.py"


class TokenBackend:
    dim = 512
    tokens_per_member = 50
    ensemble_size = 10
    tokens_per_image = tokens_per_member * ensemble_size

    def __init__(self, device="cuda"):
        spec = importlib.util.spec_from_file_location("_chexzero_token_native", NATIVE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.native = module.create_adapter(device)
        self.device = self.native.device
        self.class_names = None
        if len(self.native.models) != self.ensemble_size:
            raise RuntimeError("All ten released checkpoint members are required")
        if any(model.visual.proj.shape[1] != self.dim for model in self.native.models):
            raise RuntimeError("Unexpected checkpoint image projection dimension")
        self.metadata = {
            **self.native.metadata, "aggregation_origin": "added", "dim": self.dim,
            "tokens_per_image": self.tokens_per_image, "tokens_per_member": self.tokens_per_member,
            "token_boundary": "each native ViT final ln_post then projection, CLS and all 49 patches; concatenate members along token axis",
            "aggregation": "independent per-member/per-prompt cosine softmax over 50 tokens, temperature 1; raw projected values; L2 aggregate; native positive-minus-negative cosine sigmoid then mean ten probabilities",
            "attention_temperature": 1.0, "native_score_kind": self.native.metadata["score_kind"],
            "score_transform": "torch.logit(mean of ten member binary probabilities, eps=1e-7); no clamp is reached for bounded cosine margins",
            "zero_adapter_reference": "aggregation_only", "native_aggregation": "each member CLS only",
            "adapter_sharing": "one MLP across all ten members and all classes/tokens; checkpoint embedding coordinate alignment is not assumed or enforced",
            "recommended_cache_batch_size": 16, "recommended_train_batch_size": 32,
            "cache_batch_size": 16, "train_microbatch_size": 32,
            "source_path": str(Path(__file__).resolve()), "native_adapter_path": str(NATIVE_PATH),
            "source_files": [str(NATIVE_PATH), str(ROOT / "official/model.py")],
            "python_launcher": str(ROOT / "local_setup/python_local.sh"),
            "patch_projection_note": "The native CLS-trained final normalization/projection is also applied to patch tokens.",
        }

    @torch.no_grad()
    def prepare_classes(self, class_names):
        self.class_names = tuple(class_names)
        if not self.class_names or any(not isinstance(x, str) or not x.strip() for x in self.class_names):
            raise ValueError("Nonempty ordered class names required")
        phrases = ["pleural thickening" if name.lower() == "pleural_thickening" else name for name in self.class_names]
        prompts = [f"There is {name}." for name in phrases] + [f"There is no {name}." for name in phrases]
        text_tokens = self.native.tokenize(prompts, context_length=77).to(self.device)
        self.queries = torch.stack([
            F.normalize(model.encode_text(text_tokens), dim=-1).detach().clone()
            for model in self.native.models
        ])
        self.metadata.update(class_names=list(self.class_names), prompts=prompts)

    @staticmethod
    def _visual_tokens(visual, images):
        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat((cls, x), dim=1)
        x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype))
        x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = visual.ln_post(x)
        return x @ visual.proj

    @torch.no_grad()
    def encode_images(self, paths):
        images = torch.stack([self.native._image(path) for path in paths]).to(self.device)
        members = [self._visual_tokens(model.visual, images).float() for model in self.native.models]
        tokens = torch.cat(members, dim=1)
        if tuple(tokens.shape[1:]) != (self.tokens_per_image, self.dim):
            raise RuntimeError(f"Unexpected projected token shape: {tuple(tokens.shape)}")
        return tokens.detach()

    def score_tokens(self, tokens, token_adapter=None):
        if token_adapter is not None:
            tokens = token_adapter(tokens)
        tokens = tokens.reshape(tokens.shape[0], self.ensemble_size, self.tokens_per_member, self.dim)
        weights = torch.einsum("bmnd,mqd->bmqn", F.normalize(tokens, dim=-1), self.queries).softmax(-1)
        contexts = F.normalize(torch.einsum("bmqn,bmnd->bmqd", weights, tokens), dim=-1)
        scores = torch.einsum("bmqd,mqd->bmq", contexts, self.queries)
        count = len(self.class_names)
        probability = (scores[:, :, :count] - scores[:, :, count:]).sigmoid().mean(1)
        return torch.logit(probability, eps=1e-7)

    def native_scores_from_tokens(self, tokens):
        tokens = tokens.reshape(tokens.shape[0], self.ensemble_size, self.tokens_per_member, self.dim)
        scores = torch.einsum("bmd,mqd->bmq", F.normalize(tokens[:, :, 0], dim=-1), self.queries)
        count = len(self.class_names)
        return torch.logit((scores[:, :, :count] - scores[:, :, count:]).sigmoid().mean(1), eps=1e-7)

    @torch.no_grad()
    def native_predictions(self, paths):
        return torch.as_tensor(self.native.predict(paths, self.class_names), device=self.device)

    @torch.no_grad()
    def native_scores(self, paths):
        return torch.logit(self.native_predictions(paths), eps=1e-7)

    def release_image_encoder(self):
        if self.class_names is None:
            raise RuntimeError("prepare_classes must precede encoder release")
        self.native = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
