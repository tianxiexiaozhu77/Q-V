"""Projected-query extension for frozen MedKLIP, retaining every native query."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'local_setup'))
from query_refinement_common import center_pixels, fit_directions, refine_query


class QueryAdapter:
    def __init__(self, native):
        self.native, self.device = native, native.device
        self.class_names = None
        self.refined_queries = None
        self.fit_mode = None
        self.eta = float(os.environ.get('QUERY_REFINEMENT_ETA', '0.3'))
        if not torch.isfinite(torch.tensor(self.eta)) or self.eta < 0:
            raise ValueError('QUERY_REFINEMENT_ETA must be finite and nonnegative')
        self.metadata = {
            'model': 'MedKLIP', 'eta': self.eta, 'preserve_query_norm': True,
            'support_feature': 'spatial mean of native image_encoder projected 256-dimensional memory tokens',
            'insertion': 'replace only requested diseases after disease_embedding_layer; original full75 context (76 for COVID) and decoder/classifier retained',
            'alignment_evidence': 'The original cross-attention consumes these image-memory and disease-query representations, but equal dimensions do not establish a cosine-aligned embedding space.',
            'extension_status': 'Experimental latent-query transfer, not a mathematically equivalent CLIP/AlphaRAD cosine-space correction. No learned mapping or test-label optimization.',
            'views': 'identity or center80 after native pixel preprocessing; common evaluator reverses center80 map geometry',
        }

    def check_classes(self, names):
        return self.native.check_classes(names)

    def _pixels(self, paths, view):
        images, sizes = [], []
        for path in paths:
            with Image.open(path) as im:
                sizes.append((im.height, im.width))
                images.append(self.native.transform(im.convert('RGB')))
        return center_pixels(torch.stack(images).to(self.device), view), sizes

    def _pair(self, method, paths, names, view):
        model = self.native.model
        old = model.image_encoder
        had_override = 'image_encoder' in model.__dict__
        cached = []
        def once(pixels):
            if not cached:
                cached.append((pixels, old(pixels)))
            elif not torch.equal(pixels, cached[0][0]):
                raise ValueError('Paired visual forward received unequal view pixels')
            return cached[0][1]
        model.image_encoder = once
        try:
            return {'baseline': method(paths, names, refined=False, view=view),
                    'refined': method(paths, names, refined=True, view=view)}
        finally:
            if had_override:
                model.image_encoder = old
            else:
                del model.image_encoder

    @torch.inference_mode()
    def predict_pair(self, image_paths, class_names, view='identity'):
        return self._pair(self.predict, image_paths, class_names, view)

    @torch.inference_mode()
    def grounding_pair(self, image_paths, class_names, view='identity'):
        return self._pair(self.grounding, image_paths, class_names, view)

    @contextmanager
    def _query_hook(self, names, refined):
        idx = self.native._indices(names)
        if not refined:
            yield idx
            return
        if tuple(names) != self.class_names or self.fit_mode != self.native.active_mode:
            raise ValueError('Fit this exact class order/native query mode before refined inference')
        def replace_projection(module, args, output):
            updated = output.clone()
            updated[idx] = self.refined_queries.to(output)
            return updated
        hook = self.native.model.disease_embedding_layer.register_forward_hook(replace_projection)
        try:
            yield idx
        finally:
            hook.remove()

    @torch.inference_mode()
    def fit(self, image_paths, labels, class_names, batch_size=8, task='classification'):
        idx = self.native._indices(class_names)
        self.class_names = tuple(class_names)
        self.fit_mode = self.native.active_mode
        original = self.native.model.disease_embedding_layer(self.native.model.disease_book)[idx].detach().clone()
        features = []
        for start in range(0, len(image_paths), batch_size):
            pixels, _ = self._pixels(image_paths[start:start + batch_size], 'identity')
            features.append(self.native.model.image_encoder(pixels).mean(1).cpu())
        stats = fit_directions(torch.cat(features), labels)
        self.refined_queries = refine_query(original, stats['directions'], eta=self.eta, preserve_norm=True)
        result = {k: v.tolist() if torch.is_tensor(v) else v for k, v in stats.items()}
        result.update(self.metadata)
        result.update(class_names=list(class_names), support_count=len(image_paths), task=task,
                      query_context_count=len(self.native.classes), class_query_indices=idx,
                      original_queries=original.cpu().tolist(),
                      refined_queries=self.refined_queries.cpu().tolist())
        return result

    @torch.inference_mode()
    def predict(self, image_paths, class_names, refined=False, view='identity'):
        if not refined and view == 'identity':
            return self.native.predict(image_paths, class_names)
        pixels, _ = self._pixels(image_paths, view)
        with self._query_hook(class_names, refined) as idx:
            return self.native.model(pixels).softmax(-1)[:, idx, 1].cpu().numpy()

    @torch.inference_mode()
    def grounding(self, image_paths, class_names, refined=False, view='identity'):
        if not refined and view == 'identity':
            return self.native.grounding(image_paths, class_names)
        pixels, sizes = self._pixels(image_paths, view)
        captured = []
        hook = self.native.model.decoder.register_forward_hook(lambda module, args, output: captured.append(output[1]))
        try:
            with self._query_hook(class_names, refined) as idx:
                self.native.model(pixels)
        finally:
            hook.remove()
        ws = captured[0]
        maps = ((ws[-4] + ws[-3] + ws[-2] + ws[-1]) / 4)[:, idx]
        maps = maps.reshape(len(image_paths), len(idx), 14, 14).repeat_interleave(16, -2).repeat_interleave(16, -1)
        return [torch.nn.functional.interpolate(m[None], size=size, mode='nearest')[0].cpu().numpy()
                for m, size in zip(maps, sizes)]


def create_query_adapter(native):
    return QueryAdapter(native)
