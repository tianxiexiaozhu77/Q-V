"""Frozen KAD query refinement; independent of the audited native adapter."""
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'local_setup'))
from query_refinement_common import center_pixels, fit_directions, refine_query


class QueryAdapter:
    def __init__(self, native):
        self.native = native
        self.device = native.device
        self.class_names = None
        self.original_queries = None
        self.refined_queries = None
        self.metadata = {
            'model': 'KAD', 'eta': 0.3, 'preserve_query_norm': True,
            'support_feature': 'mean of author image-encoder projected 768-dimensional patch tokens',
            'insertion': 'author text-encoder 768-dimensional MLP output, before unchanged TQN shared LayerNorm and decoder',
            'alignment_evidence': 'Author image encoder returns pooled projected tokens for image-text alignment; text projection has the same 768-dimensional representation. Original TQN and classifier remain unchanged.',
            'extension_status': 'Experimental transfer of support-derived query refinement; improvement is not assumed.',
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

    def _queries(self, names, refined):
        if not refined:
            return self.native._queries(names)
        if tuple(names) != self.class_names or self.refined_queries is None:
            raise ValueError('Fit this exact class order before refined inference')
        return self.refined_queries

    def _pair(self, method, paths, names, view):
        encoder = self.native.image_encoder
        old = encoder.forward
        had_override = 'forward' in encoder.__dict__
        cached = []
        def once(pixels):
            if not cached:
                cached.append((pixels, old(pixels)))
            elif not torch.equal(pixels, cached[0][0]):
                raise ValueError('Paired visual forward received unequal view pixels')
            return cached[0][1]
        encoder.forward = once
        try:
            return {'baseline': method(paths, names, refined=False, view=view),
                    'refined': method(paths, names, refined=True, view=view)}
        finally:
            if had_override:
                encoder.forward = old
            else:
                del encoder.forward

    @torch.inference_mode()
    def predict_pair(self, image_paths, class_names, view='identity'):
        return self._pair(self.predict, image_paths, class_names, view)

    @torch.inference_mode()
    def grounding_pair(self, image_paths, class_names, view='identity'):
        return self._pair(self.grounding, image_paths, class_names, view)

    @torch.inference_mode()
    def fit(self, image_paths, labels, class_names, batch_size=8, task='classification'):
        self.check_classes(class_names)
        self.class_names = tuple(class_names)
        self.original_queries = self.native._queries(class_names).detach().clone()
        pooled = []
        for start in range(0, len(image_paths), batch_size):
            pixels, _ = self._pixels(image_paths[start:start + batch_size], 'identity')
            _, feature = self.native.image_encoder(pixels)
            pooled.append(feature.cpu())
        stats = fit_directions(torch.cat(pooled), labels)
        self.refined_queries = refine_query(self.original_queries, stats['directions'], eta=0.3, preserve_norm=True)
        result = {k: v.tolist() if torch.is_tensor(v) else v for k, v in stats.items()}
        result.update(self.metadata)
        result.update(class_names=list(class_names), support_count=len(image_paths), task=task,
                      original_queries=self.original_queries.cpu().tolist(),
                      refined_queries=self.refined_queries.cpu().tolist())
        return result

    @torch.inference_mode()
    def predict(self, image_paths, class_names, refined=False, view='identity'):
        if not refined and view == 'identity':
            return self.native.predict(image_paths, class_names)
        pixels, _ = self._pixels(image_paths, view)
        features, _ = self.native.image_encoder(pixels)
        return self.native.model(features, self._queries(class_names, refined)).softmax(-1)[:, :, 1].cpu().numpy()

    @torch.inference_mode()
    def grounding(self, image_paths, class_names, refined=False, view='identity'):
        if not refined and view == 'identity':
            return self.native.grounding(image_paths, class_names)
        import cv2
        pixels, sizes = self._pixels(image_paths, view)
        features, _ = self.native.image_encoder(pixels)
        _, attention = self.native.model(features, self._queries(class_names, refined), return_atten=True)
        side = int(attention.shape[-1] ** .5)
        if side * side != attention.shape[-1]:
            raise ValueError('Non-square KAD attention grid')
        maps = attention.reshape(len(image_paths), len(class_names), side, side).cpu().numpy()
        return [np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) for m in row])
                for row, (h, w) in zip(maps, sizes)]


def create_query_adapter(native):
    return QueryAdapter(native)
