"""Current-batch native pixel cache; encoder/query/scoring functions unchanged."""
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image

from adapter import Adapter as UncachedAdapter


class Adapter(UncachedAdapter):
    def __init__(self, device='cuda'):
        super().__init__(device)
        self._pixel_key = None
        self._pixel_value = None
        self._pool = ThreadPoolExecutor(max_workers=4)
        self.pixel_cache_checks = []
        self._checked_contexts = set()
        self.metadata['pixel_cache'] = 'Only the current ordered batch; native PIL/Compose unchanged; original and center views share native pixels; first batch equals serial native preprocessing bitwise.'

    def _decode(self, path):
        with Image.open(path) as image:
            return self.transform(image.convert('RGB')), (image.height, image.width)

    def prepared_pixels(self, paths):
        key = tuple(paths)
        if key != self._pixel_key:
            decoded = list(self._pool.map(self._decode, key))
            pixels = torch.stack([value[0] for value in decoded]).to(self.device)
            sizes = [value[1] for value in decoded]
            if not self.pixel_cache_checks:
                serial = torch.stack([self._decode(path)[0] for path in key]).to(self.device)
                if not torch.equal(pixels, serial):
                    raise RuntimeError('Threaded native preprocessing differs from original serial transform')
                self.pixel_cache_checks.append({'passed': True, 'images': len(key),
                    'native_preprocessing_bitwise_equal': True, 'max_abs_error': 0.0})
            self._pixel_key = key
            self._pixel_value = pixels, sizes
        return self._pixel_value

    @torch.inference_mode()
    def predict(self, image_paths, class_names):
        idx = self._indices(class_names)
        pixels, _ = self.prepared_pixels(image_paths)
        value = self.model(pixels).softmax(-1)[:, idx, 1].cpu().numpy()
        if self.active_mode not in self._checked_contexts:
            # Invoke the actual pre-cache native adapter implementation directly.
            expected = super().predict(image_paths, class_names)
            if not (value == expected).all():
                import numpy as np
                np.testing.assert_allclose(value, expected, atol=2e-6, rtol=1e-5)
            self.pixel_cache_checks.append({'passed': True, 'context': self.active_mode,
                'images': len(image_paths), 'native_score_max_abs_error': float(abs(value - expected).max())})
            self._checked_contexts.add(self.active_mode)
        return value

    def integrity(self):
        result = super().integrity()
        result['pixel_cache_checks'] = self.pixel_cache_checks
        result['pixel_cache_scope'] = 'One ordered batch; next batch replaces both pixels and sizes.'
        return result
