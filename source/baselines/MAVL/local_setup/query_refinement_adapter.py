"""Frozen MAVL support-query extension, preserving all nine disease concepts."""
from contextlib import contextmanager
from pathlib import Path
import csv
import hashlib
import json
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
        self.metadata = {
            'model': 'MAVL', 'eta': 0.3, 'preserve_query_norm': True,
            'support_feature': 'author image_encoder global attention-pooled 256-dimensional image vector',
            'insertion': 'after disease_embedding_layer; apply one support-derived disease direction equally to its nine original concept queries; retain their individual original norms',
            'alignment_evidence': 'Author forward explicitly uses normalized global image vectors and projected concept queries in cosine matching. Classification/grounding decoders remain native.',
            'concept_policy': 'One disease-level direction broadcast uniformly over all9concepts, chosen before evaluation because support labels are disease-level and provide no concept annotations. No test-based concept selection.',
            'query_context': 'Retain all75 diseases (76 for COVID), all9concepts, original normal/clear references, decoder and output heads.',
            'extension_status': 'Experimental multi-concept transfer of the shared query-refinement rule; benefit is not assumed.',
            'views': 'identity or center80 after native pixel preprocessing; common evaluator reverses center80 map geometry',
        }

    def check_classes(self, names):
        return self.native.check_classes(names)

    def set_dataset(self, entry):
        return self.native.set_dataset(entry)

    def _activate(self, names, task):
        self.check_classes(names)
        aliases = self.native.metadata['label_mapping']
        keys = [aliases.get(n.lower(), n.lower()) for n in names]
        covid = 'covid19' in keys
        if task == 'classification':
            mode = 'covid' if covid else 'feature'
        else:
            mode = 'ground_covid' if covid else 'ground'
        self.native._setup(mode)
        return [self.native.classes.index(k) for k in keys]

    def _prepare_rsna_support(self, paths, labels):
        if self.native.dataset != 'rsna_pneumonia':
            return
        # The native adapter's lookup initially covers only held-out test rows.
        # Add support lookup entries from the exact same frozen native inventory,
        # never from a new split and never by loose substring matching.
        medical = Path(__file__).resolve().parents[3]
        inventory = medical / 'RadZero_i2cl/local_assets/splits/native_public5_local_seed42/RSNA/inventory.csv'
        with inventory.open(newline='') as f:
            rows = list(csv.DictReader(f))
        by_patient = {row['patient_id']: row for row in rows}
        if len(by_patient) != len(rows):
            raise ValueError('Nonunique frozen RSNA patient IDs')
        records = []
        for path, label in zip(paths, labels):
            pid = Path(path).stem
            if pid not in by_patient:
                raise ValueError('RSNA support absent from frozen native inventory: ' + pid)
            row = by_patient[pid]
            if list(map(int, label)) != list(map(int, json.loads(row['labels_json']))):
                raise ValueError('RSNA support label differs from frozen native inventory')
            native_path = medical / 'data/radzero/prepared' / row['image_path']
            if not native_path.is_file():
                raise FileNotFoundError(native_path)
            self.native.native_images[str(path)] = str(native_path)
            records.append({'image': str(path), 'patient_id': pid, 'native_dicom': str(native_path)})
        self.metadata['support_native_remap'] = {
            'inventory': str(inventory), 'inventory_sha256': hashlib.sha256(inventory.read_bytes()).hexdigest(),
            'match': 'exact PatientID; labels checked against frozen inventory; DICOM PatientID checked again by native image reader',
            'rows': records,
        }

    def _pixels(self, paths, view):
        images, sizes = [], []
        for path in paths:
            path = str(path)
            with Image.open(path) as im:
                sizes.append((im.height, im.width))
            images.append(self.native.transform(self.native._read_image(path)))
        return center_pixels(torch.stack(images).to(self.device), view), sizes

    def _pair(self, method, paths, names, view, task):
        self._activate(names, task)
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
            result = {'baseline': method(paths, names, refined=False, view=view),
                      'refined': method(paths, names, refined=True, view=view)}
            if self.native.model is not model:
                raise ValueError('Native checkpoint changed during paired prediction')
            return result
        finally:
            if had_override:
                model.image_encoder = old
            else:
                del model.image_encoder

    @torch.inference_mode()
    def predict_pair(self, image_paths, class_names, view='identity'):
        return self._pair(self.predict, image_paths, class_names, view, 'classification')

    @torch.inference_mode()
    def grounding_pair(self, image_paths, class_names, view='identity'):
        return self._pair(self.grounding, image_paths, class_names, view, 'grounding')

    @contextmanager
    def _query_hook(self, names, refined, task):
        idx = self._activate(names, task)
        if not refined:
            yield idx
            return
        if tuple(names) != self.class_names or self.fit_mode != self.native.active_mode:
            raise ValueError('Fit this exact class order/native checkpoint mode before refined inference')
        def replace_projection(module, args, output):
            n_concepts = self.native.model.n_concepts
            updated = output.clone().reshape(len(self.native.classes), n_concepts, -1)
            updated[idx] = self.refined_queries.to(output)
            return updated.reshape_as(output)
        hook = self.native.model.disease_embedding_layer.register_forward_hook(replace_projection)
        try:
            yield idx
        finally:
            hook.remove()

    @torch.inference_mode()
    def fit(self, image_paths, labels, class_names, batch_size=8, task='classification'):
        idx = self._activate(class_names, task)
        self.class_names = tuple(class_names)
        self.fit_mode = self.native.active_mode
        self._prepare_rsna_support(image_paths, labels)
        model = self.native.model
        if model.n_concepts != 9:
            raise ValueError('Expected native nine-concept MAVL')
        original = model.disease_embedding_layer(model.concept_book).reshape(len(self.native.classes), 9, -1)[idx].detach().clone()
        features = []
        for start in range(0, len(image_paths), batch_size):
            pixels, _ = self._pixels(image_paths[start:start + batch_size], 'identity')
            _, pooled = model.image_encoder(pixels)
            features.append(pooled.cpu())
        stats = fit_directions(torch.cat(features), labels)
        self.refined_queries = refine_query(original, stats['directions'][:, None, :], eta=0.3, preserve_norm=True)
        result = {k: v.tolist() if torch.is_tensor(v) else v for k, v in stats.items()}
        result.update(self.metadata)
        result.update(class_names=list(class_names), support_count=len(image_paths), task=task,
                      active_mode=self.fit_mode, query_context_count=len(self.native.classes) * 9,
                      class_query_indices=idx, original_queries=original.cpu().tolist(),
                      refined_queries=self.refined_queries.cpu().tolist())
        return result

    @torch.inference_mode()
    def predict(self, image_paths, class_names, refined=False, view='identity'):
        if not refined and view == 'identity':
            return self.native.predict(image_paths, class_names)
        pixels, _ = self._pixels(image_paths, view)
        with self._query_hook(class_names, refined, 'classification') as idx:
            pred_class, _, _, pred_global, _ = self.native.model(pixels)
        if self.native.active_mode == 'feature':
            return pred_class.softmax(-1)[:, idx, 1].cpu().numpy()
        references = []
        for normal_idx in [0, 1]:
            normal = pred_global[:, :, [normal_idx]].repeat(1, 1, len(self.native.classes))
            references.append(torch.stack([normal, pred_global], dim=-1).softmax(-1))
        return torch.stack(references).mean(0)[:, :, idx, 1].mean(1).cpu().numpy()

    @torch.inference_mode()
    def grounding(self, image_paths, class_names, refined=False, view='identity'):
        if not refined and view == 'identity':
            return self.native.grounding(image_paths, class_names)
        pixels, sizes = self._pixels(image_paths, view)
        with self._query_hook(class_names, refined, 'grounding') as idx:
            _, ws = self.native.model(pixels)
        ws = (ws[-4] + ws[-3] + ws[-2] + ws[-1]) / 4
        side = int(ws.shape[-1] ** .5)
        if side * side != ws.shape[-1]:
            raise ValueError('Non-square MAVL attention grid')
        maps = ws.reshape(len(image_paths), len(self.native.classes), 9, -1)[:, :, [0, 6, 7, 8]].mean(2)
        maps = maps[:, idx].reshape(len(image_paths), len(idx), side, side)
        maps = torch.nn.functional.interpolate(maps, size=(224, 224), mode='bilinear', align_corners=True)
        return [torch.nn.functional.interpolate(m[None], size=size, mode='bilinear', align_corners=True)[0].cpu().numpy()
                for m, size in zip(maps, sizes)]


def create_query_adapter(native):
    return QueryAdapter(native)
