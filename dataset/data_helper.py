import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor
import torch
from collections import defaultdict


IMPRESSION_KEYWORDS = [
    'impression',
    'suggest', 'suggesting', 'suggests',
    'consistent with',
    'compatible with',
    'concerning for',
    'likely', 'most likely', 'probably',
    'no acute',
    'represents', 'represent',
    'recommend', 'recommendation',
    'rule out',
    'consider', 'considered',
    'overall',
    'in summary',
    'conclusion',
    'differential diagnosis',
]


def split_findings_impression(report,
                              strategy='heuristic',
                              ratio=0.7):
    if not report or not report.strip():
        return "", ""

    sentences = [s.strip() for s in report.split('.') if s.strip()]
    if len(sentences) == 0:
        return report, report

    if len(sentences) == 1:
        only = sentences[0]
        return only + ' .', only + ' .'

    if strategy == 'duplicate':
        f_text = ' . '.join(sentences) + ' .'
        return f_text, f_text

    if strategy == 'ratio':
        ratio = float(ratio) if ratio is not None else 0.7
        ratio = min(max(ratio, 0.1), 0.9)
        split_idx = max(1, int(len(sentences) * ratio))
        findings_sents = sentences[:split_idx]
        impression_sents = sentences[split_idx:]
        if len(impression_sents) == 0:
            impression_sents = [findings_sents[-1]]
        f_text = ' . '.join(findings_sents) + ' .'
        i_text = ' . '.join(impression_sents) + ' .'
        return f_text, i_text

    findings_sents = []
    impression_sents = []
    found_marker = False
    for sent in sentences:
        s_low = sent.lower()
        is_impression_like = any(kw in s_low for kw in IMPRESSION_KEYWORDS)
        if is_impression_like:
            found_marker = True
            impression_sents.append(sent)
        elif found_marker:
            impression_sents.append(sent)
        else:
            findings_sents.append(sent)

    if len(impression_sents) == 0:
        split_idx = max(1, int(len(sentences) * ratio))
        findings_sents = sentences[:split_idx]
        impression_sents = sentences[split_idx:]
        if len(impression_sents) == 0:
            impression_sents = [findings_sents[-1]]

    if len(findings_sents) == 0:
        findings_sents = [impression_sents[0]]

    findings_text = ' . '.join(findings_sents) + ' .'
    impression_text = ' . '.join(impression_sents) + ' .'
    return findings_text, impression_text


def _to_str_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if x is not None and str(x).strip()]
    return []


DEFAULT_FINDING_CATEGORIES = [
    'effusion', 'atelectasis', 'opacity', 'nodule', 'mass',
    'consolidation', 'infiltrate', 'pneumonia', 'edema',
    'pneumothorax', 'cardiomegaly', 'calcification', 'fibrosis',
    'pleural_thickening', 'granuloma', 'scarring', 'emphysema',
    'cavitation', 'bronchiectasis', 'hilum_adenopathy'
]


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)

        # alignment
        self.use_alignment = getattr(args, 'use_alignment', True)
        self.extract_phrases_for_alignment = getattr(args, 'extract_phrases_for_alignment', True)

        # cross-attention
        self.use_cross_attention = getattr(args, 'use_cross_attention', True)
        self.finding_categories = getattr(args, 'finding_categories', DEFAULT_FINDING_CATEGORIES)

        # Reason
        self.use_cot = getattr(args, 'use_cot', True)
        self.cot_split_strategy = getattr(args, 'cot_split_strategy', 'heuristic')
        self.cot_split_ratio = getattr(args, 'cot_split_ratio', 0.7)

        # ABN (priorrg_mimic_abn) 
        self.use_prior = getattr(args, 'use_prior', True)
        self.use_second_recent_prior = getattr(args, 'use_second_recent_prior', True)
        self.max_prior_images = getattr(args, 'max_prior_images', 2)
        self.use_factual_serialization = getattr(args, 'use_factual_serialization', True)
        self.use_abn_meta_context = getattr(args, 'use_abn_meta_context', False)
        self.merge_prior_phrases = getattr(args, 'merge_prior_phrases', False)

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values

    def _load_one_image(self, image_path, skip_missing=False):
        full_path = os.path.join(self.args.base_dir, image_path)
        if skip_missing and not os.path.exists(full_path):
            return None
        with Image.open(full_path) as pil:
            array = np.array(pil, dtype=np.uint8)
            if len(array.shape) != 3 or array.shape[-1] != 3:
                array = np.array(pil.convert("RGB"), dtype=np.uint8)
            return self._parse_image(array)

    def clean_report(self, report):
        if not report or not str(report).strip():
            return ""

        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
                .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
                .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub(r'[.,?;*!%^&_+():\-\[\]{}]', '',
                                            t.replace('"', '').replace('/', '')
                                             .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != '']
            report = ' . '.join(tokens) + ' .'
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub(r'[.,?;*!%^&_+()\[\]{}]', '',
                                            t.replace('"', '').replace('/', '')
                                             .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != '']
            report = ' . '.join(tokens) + ' .'
        return report

    def _is_valid_meta(self, text):
        if not text:
            return False
        t = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip().lower()
        if not t:
            return False
        return t not in ("none", "n/a", "na", "no", "nothing")

    def _build_abn_report(self, features):

        parts = []

        if self.use_abn_meta_context:
            ind = (features.get("indication_pure") or "").strip()
            if self._is_valid_meta(ind):
                parts.append(f"indication : {ind}")
            comp = (features.get("comparison") or "").strip()
            if self._is_valid_meta(comp):
                parts.append(f"comparison : {comp}")

        findings = (features.get("findings") or "").strip()
        impression = (features.get("impression") or "").strip()

        if findings:
            parts.append(findings)
        if impression:
            parts.append(impression)

        return " ".join(p if p.endswith(".") else p + " ." for p in parts)


    def _parse_one_prior_entry(self, entry):
        if not entry or not isinstance(entry, dict):
            return None

        img_paths = _to_str_list(entry.get("image_path"))
        view_positions = _to_str_list(entry.get("view_position"))

        f_raw = (entry.get("findings") or "").strip()
        i_raw = (entry.get("impression") or "").strip()
        full_raw = " ".join(x for x in [f_raw, i_raw] if x)

        f_clean = self.clean_report(f_raw) if f_raw else ""
        i_clean = self.clean_report(i_raw) if i_raw else ""
        text_clean = self.clean_report(full_raw)

        fs = entry.get("findings_factual_serialization", []) or []
        if not isinstance(fs, list):
            fs = []
        fs = [str(p).strip().lower() for p in fs if str(p).strip()]

        if not img_paths and not text_clean and not fs:
            return None

        return {
            'image_paths': img_paths,
            'view_positions': view_positions,
            'findings_raw': f_raw,
            'impression_raw': i_raw,
            'findings_clean': f_clean,
            'impression_clean': i_clean,
            'text_clean': text_clean,
            'factual_serialization': fs,
            'indication': (entry.get("indication_pure") or "").strip(),
            'history': (entry.get("history_pure") or "").strip(),
        }

    def _parse_prior_study(self, features):
        out = {
            'has_prior': False,
            'has_latest': False,
            'has_second_recent': False,
            'prior_image': [],
            'prior_view_positions': [],
            'prior_text': "",
            'prior_findings_text': "",
            'prior_impression_text': "",
            'prior_phrases': [],
            'prior_text_second': "",
            'prior_findings_text_second': "",
            'prior_impression_text_second': "",
            'prior_phrases_second': [],
        }

        prior = features.get("prior_study")
        if not prior or not isinstance(prior, dict):
            return out

        # ---- latest_study ----
        latest_parsed = self._parse_one_prior_entry(prior.get("latest_study"))
        if latest_parsed is not None:
            out['has_latest'] = True
            out['prior_text'] = latest_parsed['text_clean']
            out['prior_findings_text'] = latest_parsed['findings_clean']
            out['prior_impression_text'] = latest_parsed['impression_clean']
            out['prior_phrases'] = latest_parsed['factual_serialization']
            out['prior_view_positions'].extend(latest_parsed['view_positions'])

            # image
            for p in latest_parsed['image_paths']:
                if len(out['prior_image']) >= self.max_prior_images:
                    break
                try:
                    img = self._load_one_image(p, skip_missing=True)
                    if img is not None:
                        out['prior_image'].append(img)
                except Exception as e:
                    print(f"[warn] Failed to read latest prior image {p}: {e}")

        if self.use_second_recent_prior:
            second_parsed = self._parse_one_prior_entry(prior.get("second_recent_study"))
            if second_parsed is not None:
                out['has_second_recent'] = True
                out['prior_text_second'] = second_parsed['text_clean']
                out['prior_findings_text_second'] = second_parsed['findings_clean']
                out['prior_impression_text_second'] = second_parsed['impression_clean']
                out['prior_phrases_second'] = second_parsed['factual_serialization']
                out['prior_view_positions'].extend(second_parsed['view_positions'])

                for p in second_parsed['image_paths']:
                    if len(out['prior_image']) >= self.max_prior_images:
                        break
                    try:
                        img = self._load_one_image(p, skip_missing=True)
                        if img is not None:
                            out['prior_image'].append(img)
                    except Exception as e:
                        print(f"[warn] Failed to read second_recent prior image {p}: {e}")

        out['has_prior'] = out['has_latest'] or out['has_second_recent']
        return out

    # Findings
    def _get_finding_variants(self, finding):
        variants = {
            'effusion': ['effusion', 'pleural effusion', 'pleural effusions', 'effusions'],
            'atelectasis': ['atelectasis', 'atelectatic', 'subsegmental atelectasis', 'linear atelectasis'],
            'opacity': ['opacity', 'opacities', 'opacification', 'patchy opacity', 'reticular opacity'],
            'nodule': ['nodule', 'nodules', 'nodular', 'pulmonary nodule', 'lung nodule'],
            'mass': ['mass', 'masses', 'mediastinal mass', 'hilar mass'],
            'consolidation': ['consolidation', 'consolidations', 'consolidated', 'airspace consolidation'],
            'infiltrate': ['infiltrate', 'infiltrates', 'infiltration', 'interstitial infiltrate'],
            'pneumonia': ['pneumonia', 'pneumonic', 'pneumonitis', 'bronchopneumonia'],
            'edema': ['edema', 'pulmonary edema', 'interstitial edema', 'flash edema'],
            'pneumothorax': ['pneumothorax', 'pneumothoraces', 'tension pneumothorax'],
            'cardiomegaly': ['cardiomegaly', 'enlarged heart', 'cardiac enlargement', 'cardiomegaly is present'],
            'calcification': ['calcification', 'calcifications', 'calcified', 'dystrophic calcification'],
            'fibrosis': ['fibrosis', 'fibrotic', 'pulmonary fibrosis', 'interstitial fibrosis'],
            'pleural_thickening': ['pleural thickening', 'thickened pleura', 'pleural scar', 'pleural plaque'],
            'granuloma': ['granuloma', 'granulomas', 'calcified granuloma', 'granulomatous'],
            'scarring': ['scarring', 'scar', 'parenchymal scar', 'scar tissue'],
            'emphysema': ['emphysema', 'emphysematous', 'bullous emphysema', 'centrilobular emphysema'],
            'cavitation': ['cavitation', 'cavitary', 'cavity', 'cavitary lesion'],
            'bronchiectasis': ['bronchiectasis', 'bronchiectatic', 'cylindrical bronchiectasis'],
            'hilum_adenopathy': ['hilar adenopathy', 'hilum adenopathy', 'hilar lymphadenopathy', 'enlarged hilum']
        }
        return variants.get(finding, [finding])

    def extract_finding_labels(self, report_text):
        report_lower = report_text.lower()
        labels = []
        negative_patterns = [
            r'no\s+{}', r'without\s+{}', r'free of\s+{}',
            r'no evidence of\s+{}', r'absence of\s+{}'
        ]
        for finding in self.finding_categories:
            finding_variants = self._get_finding_variants(finding)
            found = False
            for variant in finding_variants:
                if variant in report_lower:
                    is_negative = False
                    for neg_pattern in negative_patterns:
                        neg_regex = neg_pattern.format(re.escape(variant))
                        if re.search(neg_regex, report_lower):
                            is_negative = True
                            break
                    if not is_negative:
                        found = True
                        break
            labels.append(1.0 if found else 0.0)
        return torch.tensor(labels, dtype=torch.float32)

    # Alignment
    def extract_medical_phrases(self, report_text):
        if not report_text:
            return []
        report_text = report_text.lower()
        phrases = []
        patterns = [
            r'(no|with|without|evidence of|demonstrating|showing|there is|there are)\s+(\w+\s+\w+)',
            r'(\w+\s+(effusion|atelectasis|opacity|nodule|consolidation|infiltrate|pneumonia|edema|cavitation|mass|calcification|fibrosis|pleural|pneumothorax))',
            r'(\w+\s+\w+\s+(effusion|atelectasis|opacity|nodule))',
            r'(right|left|bilateral|bibasal|apical|basilar|central|peripheral)\s+(\w+\s+\w+)',
            r'(upper|middle|lower|lingula|hilum|mediastinal)\s+(\w+\s+\w+)',
            r'(right\s+upper|right\s+middle|right\s+lower|left\s+upper|left\s+lower)\s+(\w+)',
            r'(\w+\s+\w+\s+\w+)',
        ]
        medical_keywords = [
            'effusion', 'atelectasis', 'opacity', 'nodule', 'consolidation',
            'infiltrate', 'pneumonia', 'edema', 'cavitation', 'mass',
            'calcification', 'fibrosis', 'pleural', 'pneumothorax',
            'cardiomegaly', 'enlargement', 'calcified', 'granuloma'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, report_text)
            for match in matches:
                if isinstance(match, tuple):
                    phrase = ' '.join(match[-2:]) if len(match) > 1 else match[0]
                else:
                    phrase = match
                words = phrase.split()
                if 2 <= len(words) <= 8 and len(phrase) > 5:
                    contains_keyword = any(keyword in phrase for keyword in medical_keywords)
                    if contains_keyword or len(phrases) < 15:
                        phrases.append(phrase)
        phrases = list(dict.fromkeys(phrases))
        phrases.sort(key=lambda x: sum(1 for kw in medical_keywords if kw in x), reverse=True)
        max_phrases = getattr(self.args, 'max_phrases_per_report', 15)
        return phrases[:max_phrases]

    def create_alignment_data(self, report_text, features=None):

        max_phrases = getattr(self.args, 'max_phrases_per_report', 15)

        if (
            self.dataset == "abn"
            and self.use_factual_serialization
            and features is not None
        ):
            fs = features.get("findings_factual_serialization", None)
            if isinstance(fs, list) and len(fs) > 0:
                simple_phrases = [str(p).strip().lower() for p in fs if str(p).strip()]

                if self.merge_prior_phrases:
                    prior = features.get("prior_study")
                    if isinstance(prior, dict):
                        for key in ('latest_study', 'second_recent_study'):
                            entry = prior.get(key)
                            if isinstance(entry, dict):
                                p_fs = entry.get("findings_factual_serialization", []) or []
                                if isinstance(p_fs, list):
                                    simple_phrases.extend(
                                        str(p).strip().lower() for p in p_fs if str(p).strip()
                                    )

                simple_phrases = list(dict.fromkeys(simple_phrases))[:max_phrases]
                return {
                    'phrases': [],
                    'simple_phrases': simple_phrases,
                    'anatomy_regions': [],
                    'findings': [],
                    'report_clean': self.clean_report(report_text),
                }

        return {
            'phrases': [],
            'simple_phrases': self.extract_medical_phrases(report_text),
            'anatomy_regions': [],
            'findings': [],
            'report_clean': self.clean_report(report_text),
        }

    def _gather_image_paths(self, features):
        image_paths = []

        if 'image_path' in features:
            image_paths = features['image_path']
        else:
            if (
                'anchor_scan' in features
                and isinstance(features['anchor_scan'], dict)
                and 'image_path' in features['anchor_scan']
            ):
                anchor_paths = features['anchor_scan']['image_path']
                if isinstance(anchor_paths, str):
                    anchor_paths = [anchor_paths]
                image_paths.extend(anchor_paths)

            if (
                'auxiliary_references' in features
                and isinstance(features['auxiliary_references'], dict)
                and 'image_path' in features['auxiliary_references']
            ):
                aux_paths = features['auxiliary_references']['image_path']
                if isinstance(aux_paths, str):
                    aux_paths = [aux_paths]
                image_paths.extend(aux_paths)

        if isinstance(image_paths, str):
            image_paths = [image_paths]
        elif isinstance(image_paths, (list, tuple)):
            image_paths = list(image_paths)
        else:
            raise TypeError(
                f"Unsupported image_path type: {type(image_paths)}, "
                f"value: {image_paths}"
            )

        if len(image_paths) == 0:
            raise KeyError(
                f"Missing image_path. Available keys: {list(features.keys())}. "
                f"Sample: {features}"
            )

        return image_paths


    def parse(self, features):
        to_return = {'id': features['id']}

        if self.dataset == "abn":
            raw_report = features.get("report") or self._build_abn_report(features)
        else:
            raw_report = features.get("report", "")

        report_clean = self.clean_report(raw_report)
        to_return['input_text'] = report_clean

        #alignment
        if self.use_alignment and self.extract_phrases_for_alignment:
            alignment_data = self.create_alignment_data(raw_report, features=features)
            to_return['alignment_data'] = alignment_data
            to_return['extracted_phrases'] = alignment_data['simple_phrases']

        #finding labels
        if self.use_cross_attention:
            finding_labels = self.extract_finding_labels(report_clean)
            to_return['finding_labels'] = finding_labels
            to_return['finding_categories'] = self.finding_categories

        #reason
        if self.use_cot:
            findings_field = features.get("findings", None)
            impression_field = features.get("impression", None)

            f_text, i_text = "", ""
            if findings_field and impression_field:
                f_text = self.clean_report(findings_field)
                i_text = self.clean_report(impression_field)
            elif findings_field and not impression_field:

                f_text = self.clean_report(findings_field)
                _, i_text = split_findings_impression(
                    f_text,
                    strategy=self.cot_split_strategy,
                    ratio=self.cot_split_ratio,
                )
                if not i_text:
                    i_text = f_text  
            elif impression_field and not findings_field:
                i_text = self.clean_report(impression_field)
                f_text = i_text
            else:
                f_text, i_text = split_findings_impression(
                    report_clean,
                    strategy=self.cot_split_strategy,
                    ratio=self.cot_split_ratio,
                )
            to_return['findings_text'] = f_text
            to_return['impression_text'] = i_text

        # ---- ABN: prior_study ----
        if self.dataset == "abn" and self.use_prior:
            prior_info = self._parse_prior_study(features)
            # has flags
            to_return['has_prior'] = prior_info['has_prior']
            to_return['has_latest_prior'] = prior_info['has_latest']
            to_return['has_second_recent_prior'] = prior_info['has_second_recent']
            # images & views
            to_return['prior_image'] = prior_info['prior_image']
            to_return['prior_view_positions'] = prior_info['prior_view_positions']
            # latest text
            to_return['prior_text'] = prior_info['prior_text']
            to_return['prior_findings_text'] = prior_info['prior_findings_text']
            to_return['prior_impression_text'] = prior_info['prior_impression_text']
            to_return['prior_phrases'] = prior_info['prior_phrases']
            # second_recent text
            to_return['prior_text_second'] = prior_info['prior_text_second']
            to_return['prior_findings_text_second'] = prior_info['prior_findings_text_second']
            to_return['prior_impression_text_second'] = prior_info['prior_impression_text_second']
            to_return['prior_phrases_second'] = prior_info['prior_phrases_second']

        if self.dataset == "abn":
            to_return['subject_id'] = features.get("subject_id", "")
            to_return['study_id'] = features.get("study_id", "")
            to_return['indication'] = (features.get("indication_pure") or "").strip()
            to_return['indication_raw'] = (features.get("indication") or "").strip()
            to_return['history'] = (features.get("history_pure") or "").strip()
            to_return['history_raw'] = (features.get("history") or "").strip()
            to_return['comparison'] = (features.get("comparison") or "").strip()
            to_return['examination'] = (features.get("examination") or "").strip()
            to_return['technique'] = (features.get("technique") or "").strip()
            to_return['recommendations'] = (features.get("recommendations") or "").strip()

            view_positions = []
            if isinstance(features.get("anchor_scan"), dict):
                view_positions.extend(features["anchor_scan"].get("view_position", []) or [])
            if isinstance(features.get("auxiliary_references"), dict):
                view_positions.extend(features["auxiliary_references"].get("view_position", []) or [])
            to_return['view_positions'] = view_positions

            anchor_n = 0
            aux_n = 0
            if isinstance(features.get("anchor_scan"), dict):
                anchor_n = len(features["anchor_scan"].get("image_path", []) or [])
            if isinstance(features.get("auxiliary_references"), dict):
                aux_n = len(features["auxiliary_references"].get("image_path", []) or [])
            to_return['n_anchor_images'] = anchor_n
            to_return['n_aux_images'] = aux_n

        image_paths = self._gather_image_paths(features)
        images = []
        for image_path in image_paths:
            with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                if len(array.shape) != 3 or array.shape[-1] != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)
        to_return["image"] = images
        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)

        self.use_alignment = getattr(args, 'use_alignment', True)
        self.cache_alignment = getattr(args, 'cache_alignment', True)
        self.use_cross_attention = getattr(args, 'use_cross_attention', True)

        if self.use_alignment and self.cache_alignment and split == 'train':
            print(f"Precomputing alignment data for {split} split...")
            self.alignment_cache = {}
            for idx, item in enumerate(self.meta):
                if args.dataset == "abn":
                    raw_report = item.get("report") or self.parser._build_abn_report(item)
                else:
                    raw_report = item.get("report", "")
                try:
                    self.alignment_cache[idx] = self.parser.create_alignment_data(
                        raw_report, features=item
                    )
                except Exception as e:
                    print(f"Error creating alignment data for index {idx}: {e}")
                    self.alignment_cache[idx] = self.parser.create_alignment_data("", features=None)
            print(f"Alignment data cached for {len(self.alignment_cache)} samples")
        else:
            self.alignment_cache = None

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        result = self.parser.transform_with_parse(self.meta[index])
        if self.alignment_cache is not None and index in self.alignment_cache:
            result['alignment_data'] = self.alignment_cache[index]
            result['extracted_phrases'] = self.alignment_cache[index]['simple_phrases']
        return result


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset


def abn_collate_fn(batch):
    if len(batch) == 0:
        return {}

    out = {}
    text_keys = [
        'id', 'subject_id', 'study_id', 'input_text',
        'findings_text', 'impression_text',
        'indication', 'indication_raw', 'history', 'history_raw',
        'comparison', 'examination', 'technique', 'recommendations',
        'prior_text', 'prior_findings_text', 'prior_impression_text',
        'prior_text_second', 'prior_findings_text_second', 'prior_impression_text_second',
    ]
    list_keys = [
        'extracted_phrases', 'view_positions', 'prior_view_positions',
        'prior_phrases', 'prior_phrases_second',
        'finding_categories',
    ]
    for k in text_keys:
        if k in batch[0]:
            out[k] = [item.get(k, "") for item in batch]
    for k in list_keys:
        if k in batch[0]:
            out[k] = [item.get(k, []) for item in batch]

    flag_keys = ['has_prior', 'has_latest_prior', 'has_second_recent_prior',
                 'n_anchor_images', 'n_aux_images']
    for k in flag_keys:
        if k in batch[0]:
            out[k] = torch.tensor(
                [int(item.get(k, 0)) for item in batch], dtype=torch.long
            )

    if 'finding_labels' in batch[0]:
        out['finding_labels'] = torch.stack([item['finding_labels'] for item in batch], dim=0)

    def _pad_images(image_lists, pad_value=0.0):
        max_n = max((len(imgs) for imgs in image_lists), default=0)
        if max_n == 0:
            return None, None

        template = None
        for imgs in image_lists:
            if len(imgs) > 0:
                template = imgs[0]
                break
        if template is None:
            return None, None

        if template.dim() == 4 and template.size(0) == 1:
            shape = template.shape[1:]  # (3, H, W)
        else:
            shape = template.shape

        B = len(image_lists)
        padded = torch.zeros(B, max_n, *shape, dtype=template.dtype) + pad_value
        mask = torch.zeros(B, max_n, dtype=torch.bool)
        for b, imgs in enumerate(image_lists):
            for j, t in enumerate(imgs):
                if t.dim() == 4 and t.size(0) == 1:
                    padded[b, j] = t[0]
                else:
                    padded[b, j] = t
                mask[b, j] = True
        return padded, mask

    image_lists = [item.get('image', []) for item in batch]
    img_padded, img_mask = _pad_images(image_lists)
    if img_padded is not None:
        out['image'] = img_padded
        out['image_mask'] = img_mask

    if 'prior_image' in batch[0]:
        prior_lists = [item.get('prior_image', []) for item in batch]
        if any(len(p) > 0 for p in prior_lists):
            prior_padded, prior_mask = _pad_images(prior_lists)
            if prior_padded is not None:
                out['prior_image'] = prior_padded
                out['prior_image_mask'] = prior_mask
        else:
            out['prior_image'] = None
            out['prior_image_mask'] = None

    if 'alignment_data' in batch[0]:
        out['alignment_data'] = [item.get('alignment_data', {}) for item in batch]

    return out


def get_phrase_statistics(dataset):
    all_phrases = []
    phrase_freq = defaultdict(int)
    for i in range(min(len(dataset), 1000)):
        item = dataset[i]
        if 'extracted_phrases' in item:
            phrases = item['extracted_phrases']
            all_phrases.extend(phrases)
            for phrase in phrases:
                phrase_freq[phrase] += 1
    print(f"Total phrases: {len(all_phrases)}")
    print(f"Unique phrases: {len(phrase_freq)}")
    sorted_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    for phrase, freq in sorted_phrases:
        print(f"  {phrase}: {freq}")
    return phrase_freq


def get_finding_statistics(dataset):
    if len(dataset) == 0:
        print("Dataset is empty")
        return
    sample = dataset[0]
    if 'finding_categories' not in sample:
        print("No finding labels in dataset")
        return
    categories = sample['finding_categories']
    label_counts = defaultdict(int)
    total_samples = 0
    for i in range(min(len(dataset), 1000)):
        item = dataset[i]
        if 'finding_labels' in item:
            labels = item['finding_labels']
            total_samples += 1
            for j, label in enumerate(labels):
                if label > 0.5:
                    label_counts[categories[j]] += 1
    print(f"Total samples analyzed: {total_samples}")
    for category in categories:
        count = label_counts[category]
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {category:25s}: {count:5d} ({percentage:5.1f}%)")
    return label_counts


def get_prior_statistics(dataset):

    if len(dataset) == 0:
        print("Dataset is empty")
        return
    total = 0
    has_prior = 0
    has_latest = 0
    has_second = 0
    has_prior_image = 0
    has_prior_text = 0
    has_prior_text_second = 0
    prior_img_count = []
    n_check = min(len(dataset), 1000)
    for i in range(n_check):
        item = dataset[i]
        if 'has_prior' not in item:
            print("No prior info in dataset (not abn or use_prior=False)")
            return
        total += 1
        if item['has_prior']:
            has_prior += 1
        if item.get('has_latest_prior'):
            has_latest += 1
        if item.get('has_second_recent_prior'):
            has_second += 1
        if item.get('prior_image'):
            has_prior_image += 1
            prior_img_count.append(len(item['prior_image']))
        if item.get('prior_text'):
            has_prior_text += 1
        if item.get('prior_text_second'):
            has_prior_text_second += 1
    print(f"Samples checked: {total}")
    print(f"  has_prior:                 {has_prior} ({has_prior / total * 100:.1f}%)")
    print(f"    has_latest_prior:        {has_latest} ({has_latest / total * 100:.1f}%)")
    print(f"    has_second_recent_prior: {has_second} ({has_second / total * 100:.1f}%)")
    print(f"  has_prior_image:           {has_prior_image} ({has_prior_image / total * 100:.1f}%)")
    print(f"  has_prior_text(latest):    {has_prior_text} ({has_prior_text / total * 100:.1f}%)")
    print(f"  has_prior_text(second):    {has_prior_text_second} ({has_prior_text_second / total * 100:.1f}%)")
    if prior_img_count:
        print(f"  avg prior images / sample (when present): {sum(prior_img_count)/len(prior_img_count):.2f}")
    return {
        'has_prior': has_prior,
        'has_latest': has_latest,
        'has_second_recent': has_second,
        'has_prior_image': has_prior_image,
        'has_prior_text': has_prior_text,
        'has_prior_text_second': has_prior_text_second,
        'total': total,
    }
