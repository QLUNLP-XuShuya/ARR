import os
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, SwinModel
from peft import get_peft_model, LoraConfig, TaskType
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor


class CrossAttentionRefinementLayer(nn.Module):
    """Pre-Norm cross-attn: query=finding queries, kv=visual tokens."""

    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.qn = nn.LayerNorm(dim)
        self.kn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.fn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, q, kv):
        out, _ = self.attn(self.qn(q), self.kn(kv), self.kn(kv), need_weights=False)
        q = q + out
        return q + self.ffn(self.fn(q))


class R2GenGPT(pl.LightningModule):

    PB = "Human: <Img>"
    PM = "</Img> describe this chest x-ray step by step.\nAssistant: Findings: "
    SI = "\n Impression: "
    SR = "\n Report: "

    DEFAULT_PROMPT = ("Human: <Img><ImageHere></Img> "
                      "Generate a comprehensive and detailed diagnosis report for this chest xray image."
                      " \nAssistant:")

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # ---- Vision ----
        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        if args.vis_use_lora:
            cfg = LoraConfig(r=args.vis_r, lora_alpha=args.vis_alpha,
                             target_modules=["query", "value"],
                             lora_dropout=args.lora_dropout, bias="none",
                             modules_to_save=["classifier"])
            self.visual_encoder = get_peft_model(self.visual_encoder, cfg)
            self.visual_encoder.print_trainable_parameters()
        elif args.freeze_vm:
            for _, p in self.visual_encoder.named_parameters():
                p.requires_grad = False

        # ---- LLaMA ----
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0

        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model, torch_dtype=torch.float16,
                load_in_8bit=True, device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model, torch_dtype=torch.float16,
            )

        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                             r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout)
            self.llama_model = get_peft_model(self.llama_model, cfg)
            self.llama_model.print_trainable_parameters()
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for _, p in self.llama_model.named_parameters():
                p.requires_grad = False

        self.use_grad_ckpt = bool(getattr(args, 'gradient_checkpointing', True))
        if self.use_grad_ckpt:
            try:
                self.llama_model.gradient_checkpointing_enable()
                self.llama_model.config.use_cache = False
                if hasattr(self.llama_model, 'enable_input_require_grads'):
                    self.llama_model.enable_input_require_grads()
                print('LLaMA gradient_checkpointing enabled (use_cache=False)')
            except Exception as e:
                print(f'[WARN] grad ckpt failed: {e}')
                self.use_grad_ckpt = False

        self.llama_proj = nn.Linear(self.visual_encoder.num_features,
                                    self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        self.use_alignment = getattr(args, 'use_alignment', True)
        self.align_loss_weight = getattr(args, 'align_loss_weight', 0.1)
        self.align_temperature = getattr(args, 'align_temperature', 0.07)
        self.align_projection = nn.Linear(self.visual_encoder.num_features,
                                          self.llama_model.config.hidden_size)
        self.use_medical_encoder = getattr(args, 'use_medical_encoder', False)
        if self.use_medical_encoder:
            from transformers import AutoTokenizer, AutoModel
            self.medical_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
            self.medical_encoder = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
            for p in self.medical_encoder.parameters():
                p.requires_grad = False

        self.use_cross_attention = getattr(args, 'use_cross_attention', True)
        self.num_finding_queries = getattr(args, 'num_finding_queries', 8)
        self.finding_query_dim = getattr(args, 'finding_query_dim', 256)
        self.finding_loss_weight = getattr(args, 'finding_loss_weight', 0.02)
        self.finding_categories = getattr(args, 'finding_categories', [
            'effusion', 'atelectasis', 'opacity', 'nodule', 'mass',
            'consolidation', 'infiltrate', 'pneumonia', 'edema',
            'pneumothorax', 'cardiomegaly', 'calcification', 'fibrosis',
            'pleural_thickening', 'granuloma', 'scarring', 'emphysema',
            'cavitation', 'bronchiectasis', 'hilum_adenopathy'
        ])
        if self.use_cross_attention:
            self.finding_queries = nn.Parameter(
                torch.randn(1, self.num_finding_queries, self.finding_query_dim)
            )
            nn.init.xavier_uniform_(self.finding_queries)
            self.vision_to_query = nn.Linear(self.llama_model.config.hidden_size,
                                             self.finding_query_dim)
            n_layers = max(1, getattr(args, 'cross_attention_layers', 2))
            n_heads = getattr(args, 'cross_attention_heads', 4)
            self.cross_attention_layers = nn.ModuleList([
                CrossAttentionRefinementLayer(self.finding_query_dim, n_heads, 0.1)
                for _ in range(n_layers)
            ])
            self.query_to_llm = nn.Linear(self.finding_query_dim,
                                          self.llama_model.config.hidden_size)
            self.finding_classifier = nn.Sequential(
                nn.LayerNorm(self.finding_query_dim),
                nn.Linear(self.finding_query_dim, len(self.finding_categories))
            )
            self.refine_gate = nn.Parameter(torch.tensor(0.05))

        self.use_cot = getattr(args, 'use_cot', True)
        self.cot_wf = getattr(args, 'cot_findings_loss_weight', 0.3)
        self.cot_wr = getattr(args, 'cot_reasoning_loss_weight', 0.3)
        self.cot_wp = getattr(args, 'cot_report_loss_weight', 1.0)
        self.cot_inference_mode = getattr(args, 'cot_inference_mode', True)
        self.cot_max_f = getattr(args, 'cot_max_findings_tokens', 60)
        self.cot_max_r = getattr(args, 'cot_max_reasoning_tokens', 40)
        if self.use_cot:
            print(f'CoT enabled (single-pass) | weights f={self.cot_wf}, r={self.cot_wr}, p={self.cot_wp}')

        if args.delta_file is not None:
            ckpt = torch.load(args.delta_file, map_location="cpu", weights_only=False)
            self.load_state_dict(ckpt["model"], strict=False)
            print(f'Loaded checkpoint: {args.delta_file}')

    def _set_inference_mode(self, on=True):
        if on:
            try:
                self.llama_model.gradient_checkpointing_disable()
            except Exception:
                pass
            self.llama_model.config.use_cache = True
        else:
            if self.use_grad_ckpt:
                try:
                    self.llama_model.gradient_checkpointing_enable()
                    if hasattr(self.llama_model, 'enable_input_require_grads'):
                        self.llama_model.enable_input_require_grads()
                except Exception:
                    pass
                self.llama_model.config.use_cache = False


    def _tok(self, text):
        return self.llama_tokenizer(text, add_special_tokens=False,
                                    return_tensors='pt').input_ids.to(self.device)

    def _emb(self, ids):
        return self.embed_tokens(ids)

    def score(self, ref, hypo):
        scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                   (Rouge(), "ROUGE_L"), (Meteor(), "METEOR"), (Cider(), "CIDEr")]
        out = {}
        for s, m in scorers:
            sc, _ = s.compute_score(ref, hypo)
            if isinstance(sc, list):
                for mi, si in zip(m, sc):
                    out[mi] = si
            else:
                out[m] = sc
        return out

    def decode(self, t):
        if len(t) == 0:
            return ""
        if t[0] == 0:
            t = t[1:]
        if len(t) == 0:
            return ""
        if t[0] == 1:
            t = t[1:]
        s = self.llama_tokenizer.decode(t, add_special_tokens=False)
        return s.split('</s>')[0].strip().replace('<unk>', '')


    def _stack_first_view(self, imgs_list):
        firsts = []
        for x in imgs_list:
            im = x[0] if isinstance(x, list) else x
            if im.dim() == 3:
                im = im.unsqueeze(0)
            firsts.append(im)
        try:
            return torch.cat(firsts, dim=0)
        except Exception:
            return None

    def prepare_visual_inputs(self, samples, return_raw=True):
        images = samples["image"]
        batched = self._stack_first_view(images)
        if batched is None or batched.shape[0] != len(images):
            # fallback per-sample
            embs, atts, raws = [], [], []
            for x in images:
                im = x[0] if isinstance(x, list) else x
                if im.dim() == 3:
                    im = im.unsqueeze(0)
                if self.hparams.global_only:
                    rt = self.visual_encoder(im)['pooler_output'].unsqueeze(1)
                else:
                    rt = self.visual_encoder(im)['last_hidden_state']
                e = self.llama_proj(rt)
                embs.append(e)
                atts.append(torch.ones(e.shape[:-1], dtype=torch.long, device=e.device))
                if return_raw:
                    raws.append(rt)
            img_embeds = torch.cat(embs, 0)
            atts_img = torch.cat(atts, 0)
            raw_vision_tokens = torch.cat(raws, 0) if return_raw else None
        else:
            if self.hparams.global_only:
                raw_vision_tokens = self.visual_encoder(batched)['pooler_output'].unsqueeze(1)
            else:
                raw_vision_tokens = self.visual_encoder(batched)['last_hidden_state']
            img_embeds = self.llama_proj(raw_vision_tokens)
            atts_img = torch.ones(img_embeds.shape[:-1], dtype=torch.long, device=img_embeds.device)
            if not return_raw:
                raw_vision_tokens = None

        finding_logits = None
        if self.use_cross_attention:
            img_embeds, atts_img, finding_logits = self._apply_cross_attn(img_embeds, atts_img)

        img_embeds = self.layer_norm(img_embeds)
        return img_embeds, atts_img, raw_vision_tokens, finding_logits

    def _apply_cross_attn(self, img_embeds, atts_img):
        B = img_embeds.shape[0]
        v = self.vision_to_query(img_embeds)
        q = self.finding_queries.expand(B, -1, -1)
        for layer in self.cross_attention_layers:
            q = layer(q, v)
        refined = self.query_to_llm(q) * torch.tanh(self.refine_gate)
        finding_logits = self.finding_classifier(q.mean(dim=1))
        fused = torch.cat([img_embeds, refined], dim=1)
        extra = torch.ones((B, refined.shape[1]), dtype=atts_img.dtype, device=atts_img.device)
        return fused, torch.cat([atts_img, extra], dim=1), finding_logits

    def _build_cot_inputs(self, img_embeds, findings_t, impression_t, report_t):
        device = img_embeds.device
        B, Lvis, H = img_embeds.shape
        tok = self.llama_tokenizer

        pb = self._tok(self.PB);  pb_e = self._emb(pb);  Lpb = pb_e.shape[1]
        pm = self._tok(self.PM);  pm_e = self._emb(pm);  Lpm = pm_e.shape[1]
        si = self._tok(self.SI);  si_e = self._emb(si);  Lsi = si_e.shape[1]
        sr = self._tok(self.SR);  sr_e = self._emb(sr);  Lsr = sr_e.shape[1]

        bos_id = tok.bos_token_id
        pad_id = tok.pad_token_id
        bos_e = self._emb(torch.full((1, 1), bos_id, dtype=torch.long, device=device))
        pad_e = self._emb(torch.full((1, 1), pad_id, dtype=torch.long, device=device))

        f_ids = tok(list(findings_t), add_special_tokens=False, padding=False,
                    truncation=True, max_length=self.cot_max_f).input_ids
        i_ids = tok(list(impression_t), add_special_tokens=False, padding=False,
                    truncation=True, max_length=self.cot_max_r).input_ids
        r_ids = tok([t + self.end_sym for t in report_t], add_special_tokens=False,
                    padding=False, truncation=True, max_length=self.hparams.max_length).input_ids

        unk = tok.unk_token_id if tok.unk_token_id is not None else 0
        seqs, lens = [], []
        f_pos, i_pos, r_pos = [], [], []
        for b in range(B):
            f = f_ids[b] or [unk]; i_ = i_ids[b] or [unk]; r = r_ids[b] or [unk]
            Lf, Li, Lr = len(f), len(i_), len(r)
            fe = self._emb(torch.tensor([f], dtype=torch.long, device=device))
            ie = self._emb(torch.tensor([i_], dtype=torch.long, device=device))
            re_ = self._emb(torch.tensor([r], dtype=torch.long, device=device))
            seq = torch.cat([bos_e, pb_e, img_embeds[b:b+1], pm_e,
                             fe, si_e, ie, sr_e, re_], dim=1)
            seqs.append(seq); lens.append(seq.shape[1])
            of = 1 + Lpb + Lvis + Lpm
            f_pos.append((of, of + Lf, f))
            i_pos.append((of + Lf + Lsi, of + Lf + Lsi + Li, i_))
            r_pos.append((of + Lf + Lsi + Li + Lsr,
                          of + Lf + Lsi + Li + Lsr + Lr, r))

        T = max(lens)
        embs = torch.zeros((B, T, H), dtype=seqs[0].dtype, device=device)
        attn = torch.zeros((B, T), dtype=torch.long, device=device)
        Lf_lab = torch.full((B, T), -100, dtype=torch.long, device=device)
        Li_lab = torch.full((B, T), -100, dtype=torch.long, device=device)
        Lr_lab = torch.full((B, T), -100, dtype=torch.long, device=device)
        for b in range(B):
            t = lens[b]
            embs[b, :t] = seqs[b][0]
            if t < T:
                embs[b, t:] = pad_e[0, 0]
            attn[b, :t] = 1
            for lab, pos in [(Lf_lab, f_pos[b]), (Li_lab, i_pos[b]), (Lr_lab, r_pos[b])]:
                s, e, ids = pos
                lab[b, s:e] = torch.tensor(ids, dtype=torch.long, device=device)
        return embs, attn, Lf_lab, Li_lab, Lr_lab

    @staticmethod
    def _seg_loss(logits, labels):
        sl = logits[..., :-1, :].contiguous()
        ll = labels[..., 1:].contiguous()
        if (ll != -100).sum() == 0:
            return logits.sum() * 0.0
        return F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1),
                               ignore_index=-100, reduction='mean')

    def _extract_phrases(self, txt):
        if not txt:
            return []
        txt = txt.lower()
        out = []
        pats = [
            r'(no|with|without|evidence of|demonstrating|showing|there is|there are)\s+(\w+\s+\w+)',
            r'(\w+\s+(effusion|atelectasis|opacity|nodule|consolidation|infiltrate|pneumonia|edema|cavitation|mass|calcification|fibrosis))',
            r'(right|left|bilateral|bibasal|apical|basilar|central|peripheral)\s+(\w+\s+\w+)',
            r'(upper|middle|lower|lingula|hilum|mediastinal)\s+(\w+\s+\w+)',
        ]
        for p in pats:
            for m in re.findall(p, txt):
                ph = ' '.join(m[-2:]) if isinstance(m, tuple) and len(m) > 1 else (m if isinstance(m, str) else m[0])
                if 2 <= len(ph.split()) <= 8 and len(ph) > 5:
                    out.append(ph)
        return list(dict.fromkeys(out))[:6]

    def _encode_phrases_batch(self, phrases):
        device = self.device
        if self.use_medical_encoder:
            with torch.no_grad():
                t = self.medical_tokenizer(phrases, return_tensors='pt',
                                           padding=True, truncation=True, max_length=32).to(device)
                emb = self.medical_encoder(**t).last_hidden_state[:, 0, :]
        else:
            with torch.no_grad():
                t = self.llama_tokenizer(phrases, return_tensors='pt', padding=True,
                                         truncation=True, max_length=16,
                                         add_special_tokens=False).to(device)
                full = self.embed_tokens(t.input_ids)
                m = t.attention_mask.unsqueeze(-1).float()
                emb = (full * m).sum(1) / m.sum(1).clamp(min=1)
        return emb

    def _contrastive(self, patches, phrases, T=0.07):
        if patches.shape[0] == 0 or phrases.shape[0] == 0:
            return torch.tensor(0.0, device=patches.device)
        patches = F.normalize(patches, dim=-1)
        phrases = F.normalize(phrases, dim=-1).to(patches.dtype)
        sim = patches @ phrases.T / T
        l1 = F.cross_entropy(sim, sim.detach().argmax(dim=1))
        l2 = F.cross_entropy(sim.T, sim.detach().argmax(dim=0))
        return (l1 + l2) / 2

    def compute_alignment_loss(self, vision_tokens, reports, extracted=None):
        if not self.use_alignment or not self.training:
            return vision_tokens.sum() * 0.0
        B = vision_tokens.shape[0]
        proj = self.align_projection(vision_tokens)

        all_p, owner = [], []
        for i in range(B):
            ph = (extracted[i][:6] if (extracted is not None and i < len(extracted) and extracted[i])
                  else self._extract_phrases(reports[i]))
            for p in ph:
                all_p.append(p); owner.append(i)
        if not all_p:
            return vision_tokens.sum() * 0.0

        emb = self._encode_phrases_batch(all_p)
        owner_t = torch.tensor(owner, device=vision_tokens.device)
        total = vision_tokens.sum() * 0.0
        valid = 0
        for i in range(B):
            mask = (owner_t == i)
            if not mask.any():
                continue
            total = total + self._contrastive(proj[i], emb[mask], T=self.align_temperature)
            valid += 1
        return total / max(valid, 1) if valid else total

    def compute_finding_loss(self, logits, labels):
        if logits is None:
            return torch.tensor(0.0, device=self.device)
        if labels is None:
            return torch.tensor(0.0, device=logits.device)
        return nn.BCEWithLogitsLoss()(logits, labels.to(logits.device).float())

    def forward(self, samples):
        if self.training and self.use_grad_ckpt and self.llama_model.config.use_cache:
            self._set_inference_mode(False)

        img_embeds, atts_img, raw_vt, finding_logits = self.prepare_visual_inputs(
            samples, return_raw=True
        )
        report_text = samples["input_text"]
        findings_t = samples.get("findings_text", report_text)
        impression_t = samples.get("impression_text", report_text)

        log = {}
        total = torch.zeros((), device=img_embeds.device)

        if self.use_cot:
            embs, attn, lf, li, lr = self._build_cot_inputs(
                img_embeds, findings_t, impression_t, report_text
            )
            out = self.llama_model(inputs_embeds=embs, attention_mask=attn, return_dict=True)
            logits = out.logits
            f_loss = self._seg_loss(logits, lf)
            r_loss = self._seg_loss(logits, li)
            p_loss = self._seg_loss(logits, lr)
            total = total + (self.cot_wf * f_loss + self.cot_wr * r_loss + self.cot_wp * p_loss)
            log['cot_f'] = f_loss.detach()
            log['cot_r'] = r_loss.detach()
            log['cot_p'] = p_loss.detach()
            log['lm_loss'] = p_loss.detach()
        else:
            embs, attn, tgt = self._build_default_inputs(img_embeds, atts_img, report_text)
            out = self.llama_model(inputs_embeds=embs, attention_mask=attn,
                                   return_dict=True, labels=tgt)
            total = total + out.loss
            log['lm_loss'] = out.loss.detach()

        if self.training and self.use_alignment and raw_vt is not None:
            al = self.compute_alignment_loss(raw_vt, report_text,
                                             extracted=samples.get('extracted_phrases'))
            total = total + self.align_loss_weight * al
            log['align'] = al.detach()

        if self.training and self.use_cross_attention and finding_logits is not None:
            fl = samples.get('finding_labels')
            if fl is not None:
                if isinstance(fl, list):
                    fl = torch.stack(fl)
                fcl = self.compute_finding_loss(finding_logits, fl)
                total = total + self.finding_loss_weight * fcl
                log['fcls'] = fcl.detach()

        for k, v in log.items():
            self.log(k, v, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": total}

    def training_step(self, batch, batch_idx):
        out = self(batch)
        self.log('train_loss', out['loss'], prog_bar=True, logger=True, sync_dist=True)
        return out["loss"]

    def _wrap_default(self, img_embeds):
        pb_t, pa_t = self.DEFAULT_PROMPT.split('<ImageHere>')
        B = img_embeds.shape[0]
        pb = self._emb(self._tok(pb_t)).expand(B, -1, -1)
        pa = self._emb(self._tok(pa_t)).expand(B, -1, -1)
        wrapped = torch.cat([pb, img_embeds, pa], dim=1)
        atts = torch.ones(wrapped.shape[:2], dtype=torch.long, device=wrapped.device)
        return wrapped, atts

    def _build_default_inputs(self, img_embeds, atts_img, target_texts):
        wrapped, watts = self._wrap_default(img_embeds)
        device = wrapped.device
        B = wrapped.shape[0]
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in target_texts]
        tk = self.llama_tokenizer(text, return_tensors="pt", padding="max_length",
                                  truncation=True, max_length=self.hparams.max_length,
                                  add_special_tokens=False).to(device)
        targets = tk.input_ids.masked_fill(tk.input_ids == 0, -100)
        empty = torch.full([B, watts.shape[1] + 1], -100, dtype=torch.long, device=device)
        targets = torch.cat([empty, targets], dim=1)
        bos = torch.full([B, 1], self.llama_tokenizer.bos_token_id,
                         dtype=tk.input_ids.dtype, device=device)
        bos_e = self._emb(bos)
        regress_e = self._emb(tk.input_ids)
        embs = torch.cat([bos_e, wrapped, regress_e], dim=1)
        attn = torch.cat([watts[:, :1], watts, tk.attention_mask], dim=1)
        return embs, attn, targets

    @torch.no_grad()
    def _build_inf_seq(self, img_embeds, prompts_after):
        device = img_embeds.device
        B = img_embeds.shape[0]
        tok = self.llama_tokenizer
        pad_id = tok.pad_token_id
        pb_e = self._emb(self._tok(self.PB))
        bos_e = self._emb(torch.full((1, 1), tok.bos_token_id, dtype=torch.long, device=device))
        pad_e = self._emb(torch.full((1, 1), pad_id, dtype=torch.long, device=device))

        seqs, lens = [], []
        for b in range(B):
            pa = self._emb(self._tok(prompts_after[b]))
            seq = torch.cat([bos_e, pb_e, img_embeds[b:b+1], pa], dim=1)
            seqs.append(seq); lens.append(seq.shape[1])

        T = max(lens)
        H = img_embeds.shape[-1]
        embs = torch.zeros((B, T, H), dtype=seqs[0].dtype, device=device)
        attn = torch.zeros((B, T), dtype=torch.long, device=device)
        for b in range(B):
            t = lens[b]; pad = T - t
            if pad > 0:
                embs[b, :pad] = pad_e[0, 0]
                embs[b, pad:] = seqs[b][0]
                attn[b, pad:] = 1
            else:
                embs[b] = seqs[b][0]
                attn[b] = 1
        return embs, attn

    @torch.no_grad()
    def _gen(self, embs, attn, min_new, max_new):
        out = self.llama_model.generate(
            inputs_embeds=embs, attention_mask=attn,
            num_beams=self.hparams.beam_size, do_sample=self.hparams.do_sample,
            min_new_tokens=min_new, max_new_tokens=max_new,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        return [self.decode(o) for o in out]

    @staticmethod
    def _trunc(t, mw=80):
        if not t:
            return ""
        w = t.split()
        return " ".join(w[:mw]) if len(w) > mw else t

    @torch.no_grad()
    def _cot_full_inference(self, samples):
        self._set_inference_mode(True)
        img_embeds, _, _, _ = self.prepare_visual_inputs(samples, return_raw=False)
        B = img_embeds.shape[0]

        # Step 1: findings
        pa1 = [self.PM] * B
        e, a = self._build_inf_seq(img_embeds, pa1)
        findings = self._gen(e, a, 20, self.cot_max_f)

        # Step 2: impression
        pa2 = [self.PM + self._trunc(findings[b]) + self.SI for b in range(B)]
        e, a = self._build_inf_seq(img_embeds, pa2)
        impressions = self._gen(e, a, 10, self.cot_max_r)

        # Step 3: report
        pa3 = [self.PM + self._trunc(findings[b]) + self.SI + self._trunc(impressions[b]) + self.SR
               for b in range(B)]
        e, a = self._build_inf_seq(img_embeds, pa3)
        reports = self._gen(e, a, self.hparams.min_new_tokens, self.hparams.max_new_tokens)
        return findings, impressions, reports

    @torch.no_grad()
    def _single_head_inference(self, samples):
        self._set_inference_mode(True)
        img_embeds, _, _, _ = self.prepare_visual_inputs(samples, return_raw=False)
        wrapped, watts = self._wrap_default(img_embeds)
        B = wrapped.shape[0]
        bos = torch.full([B, 1], self.llama_tokenizer.bos_token_id,
                         dtype=torch.long, device=watts.device)
        bos_e = self._emb(bos)
        embs = torch.cat([bos_e, wrapped], dim=1)
        attn = torch.cat([watts[:, :1], watts], dim=1)
        return self._gen(embs, attn, self.hparams.min_new_tokens, self.hparams.max_new_tokens)

    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        tk = self.llama_tokenizer(samples['input_text'], return_tensors="pt",
                                  padding="max_length", truncation=True,
                                  max_length=self.hparams.max_length,
                                  add_special_tokens=False).to(self.device)
        if self.use_cot and self.cot_inference_mode:
            f, i, hypo = self._cot_full_inference(samples)
        else:
            hypo = self._single_head_inference(samples)
            f = ["" for _ in hypo]; i = ["" for _ in hypo]
        ref = [self.decode(x) for x in tk['input_ids']]
        self.val_step_outputs.append({
            "hypo": hypo, "ref": ref, "id": samples["id"],
            "f_pred": f, "i_pred": i,
        })
        return hypo, ref

    def on_validation_epoch_end(self):
        ref, hypo, ids, all_f, all_i = [], [], [], [], []
        for x in self.val_step_outputs:
            ref.extend(x['ref']); hypo.extend(x['hypo']); ids.extend(x['id'])
            all_f.extend(x['f_pred']); all_i.extend(x['i_pred'])
        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        rdir = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(rdir, exist_ok=True)
        ce, gs = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(rdir, f"result_{ce}_{gs}.json"), 'w'))
        json.dump(ref, open(os.path.join(rdir, 'refs.json'), 'w'))
        if self.use_cot and self.cot_inference_mode and self.trainer.local_rank == 0:
            cot_dump = {k: {'findings': f, 'impression': im, 'report': hypo[k][0]}
                        for k, f, im in zip(ids, all_f, all_i)}
            json.dump(cot_dump, open(os.path.join(rdir, f"cot_{ce}_{gs}.json"), 'w'))
        self.print(eval_res)

        s = sum(eval_res[t] * w for t, w in zip(self.hparams.scorer_types, self.hparams.weights))
        if self.trainer.local_rank == 0 and s > self.val_score:
            self.save_checkpoint(eval_res)
            self.val_score = s
        self.val_step_outputs.clear()

        if self.training and self.use_grad_ckpt:
            self._set_inference_mode(False)

    def save_checkpoint(self, eval_res):
        ce, gs = self.trainer.current_epoch, self.trainer.global_step
        keep = {k for k, v in self.named_parameters() if v.requires_grad}
        sd = self.state_dict()
        for k in list(sd.keys()):
            if k not in keep:
                del sd[k]
        save_obj = {"model": sd, "config": self.hparams, "epoch": ce, "step": gs}
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        path = os.path.join(self.hparams.savedmodel_path, 'checkpoints',
                            f"checkpoint_epoch{ce}_step{gs}_bleu{eval_res['Bleu_4']:3f}_cider{eval_res['CIDEr']:3f}.pth")
        self.print(f"Saving checkpoint to {path}.")
        torch.save(save_obj, path)

    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        tk = self.llama_tokenizer(samples['input_text'], return_tensors="pt",
                                  padding="max_length", truncation=True,
                                  max_length=self.hparams.max_length,
                                  add_special_tokens=False).to(self.device)
        if self.use_cot and self.cot_inference_mode:
            f, i, hypo = self._cot_full_inference(samples)
        else:
            hypo = self._single_head_inference(samples)
            f = ["" for _ in hypo]; i = ["" for _ in hypo]
        ref = [self.decode(x) for x in tk['input_ids']]
        self.test_step_outputs.append({
            "hypo": hypo, "ref": ref, "id": samples["id"],
            "f_pred": f, "i_pred": i,
        })
        return hypo, ref

    def on_test_epoch_end(self):
        ref, hypo, ids, all_f, all_i = [], [], [], [], []
        for x in self.test_step_outputs:
            ref.extend(x['ref']); hypo.extend(x['hypo']); ids.extend(x['id'])
            all_f.extend(x['f_pred']); all_i.extend(x['i_pred'])
        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)
        rdir = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(rdir, exist_ok=True)
        json.dump(hypo, open(os.path.join(rdir, "test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(rdir, 'test_refs.json'), 'w'))
        if self.use_cot and self.cot_inference_mode and self.trainer.local_rank == 0:
            cot_dump = {k: {'findings': f, 'impression': im, 'report': hypo[k][0]}
                        for k, f, im in zip(ids, all_f, all_i)}
            json.dump(cot_dump, open(os.path.join(rdir, "test_cot.json"), 'w'))
        self.print(f"Test result: {eval_res}")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.learning_rate
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": sch}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)
