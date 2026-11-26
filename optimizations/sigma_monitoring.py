"""Read-only training diagnostics; existing train/loss_sigma_* EMA is untouched.

Scalar statistics cover every admitted example in the reporting window. Output
moments sample entire optimizer updates at tensor_every_n_steps. All moments are
means per example before aggregation, matching per-example velocity MSE across
different image sizes. Window means are sample-weighted, not means of update
means. They are training observations, not fixed-data or held-out validation.

record_micro performs no host reads, RNG draws, collectives, forward or backward.
end_update reduces one packed tensor and copies it to the host once per report.
"""
from __future__ import annotations

import math
import re

import torch
import torch.distributed as dist


BINS = 10
BASE_FIELDS = 6  # count, finite count, loss, loss squared, sigma, text tokens
OUTPUT_FIELDS = 12  # count, finite count, loss, p2, v2, x0², eps², p*v, p, v, x0, eps


class SigmaMonitor:
    def __init__(self, config, device):
        options = config.get('monitoring', {}).get('sigma', {})
        self.enabled = bool(config.get('monitoring', {}).get('enabled', False)
                            and options.get('enabled', False))
        self.device = torch.device(device)
        self.interval = int(options.get('every_n_steps', 100))
        self.tensor_interval = int(options.get('tensor_every_n_steps', 10))
        self.sample_outputs = False
        self.updates = self.output_updates = 0
        if not self.enabled:
            return
        if self.interval <= 0 or self.tensor_interval <= 0 or self.interval % self.tensor_interval:
            raise ValueError('Sigma reporting interval must be positive and divisible by tensor sampling interval')
        resolutions = set()
        for bucket in config.get('dataset', {}).get('buckets', []):
            name = str(bucket).split(':')[-1]
            if re.fullmatch(r'\d+_square', name):
                width = name.split('_')[0]
                resolutions.add(f'{width}x{width}')
            elif re.fullmatch(r'\d+x\d+', name):
                resolutions.add(name)
        self.groups = ['all', *[f'resolution/{name}' for name in sorted(resolutions)],
                       'resolution/other', 'cfg/kept', 'cfg/dropped',
                       'text_tokens/0-128', 'text_tokens/129-512', 'text_tokens/513-plus']
        self._group_ids = {name: i for i, name in enumerate(self.groups)}
        self._base = torch.zeros(BASE_FIELDS, len(self.groups) * BINS, device=self.device, dtype=torch.float64)
        self._output = torch.zeros(OUTPUT_FIELDS, BINS, device=self.device, dtype=torch.float64)

    def begin_update(self, step):
        self.sample_outputs = self.enabled and int(step) % self.tensor_interval == 0

    @torch.no_grad()
    def record_micro(self, sigma, losses, text_mask, cfg_dropped, image_size,
                     prediction=None, latents=None, noise=None):
        if not self.enabled:
            return
        sigma, losses = sigma.detach().float(), losses.detach().float()
        tokens = text_mask.detach().sum(dim=-1, dtype=torch.float32)
        drop = torch.as_tensor(cfg_dropped, device=self.device, dtype=torch.long)
        bins = (torch.nan_to_num(sigma, nan=0., posinf=1., neginf=0.) * BINS).long().clamp_(0, BINS - 1)
        width, height = image_size
        resolution = self._group_ids.get(f'resolution/{width}x{height}', self._group_ids['resolution/other'])
        text_group = self._group_ids['text_tokens/0-128'] + (tokens > 128).long() + (tokens > 512).long()
        groups = torch.stack((torch.zeros_like(bins), torch.full_like(bins, resolution),
                              self._group_ids['cfg/kept'] + drop, text_group))
        indices = (groups * BINS + bins).flatten()
        finite = torch.isfinite(losses) & torch.isfinite(sigma)
        safe_loss = torch.where(finite, losses, 0.).double()
        packed = torch.stack((torch.ones_like(safe_loss), finite.double(), safe_loss,
                              safe_loss.square(), torch.where(finite, sigma, 0.).double(),
                              torch.where(finite, tokens, 0.).double()))
        self._base.scatter_add_(1, indices.expand(BASE_FIELDS, -1), packed.repeat(1, groups.shape[0]))
        if not self.sample_outputs:
            return
        # After backward, so these short-lived FP32 scratch tensors do not overlap
        # with saved DiT activations. Only detached scalar moments survive.
        p = prediction.detach().float().flatten(1)
        x0, eps = latents.detach().float().flatten(1), noise.detach().float().flatten(1)
        target = eps - x0
        moments = torch.stack((losses, p.square().mean(1), target.square().mean(1),
                               x0.square().mean(1), eps.square().mean(1), (p * target).mean(1),
                               p.mean(1), target.mean(1), x0.mean(1), eps.mean(1)))
        valid = torch.isfinite(moments).all(dim=0) & finite
        values = torch.cat((torch.ones_like(losses)[None], valid.float()[None],
                            torch.where(valid[None], moments, 0.)), dim=0).double()
        self._output.scatter_add_(1, bins.expand(OUTPUT_FIELDS, -1), values)

    @staticmethod
    def _loss_metrics(row, prefix, detailed=False, total=None):
        count, valid, summed, squared, sigma, tokens = row
        payload = {f'{prefix}/samples': count}
        if count != valid:
            payload[f'{prefix}/nonfinite_samples'] = count - valid
        if not valid or count != valid:
            return payload  # do not disguise a broken bucket with a finite subset mean
        mean = summed / valid
        payload[f'{prefix}/loss_mean'] = mean
        if detailed:
            payload.update({f'{prefix}/fraction': count / total,
                            f'{prefix}/sigma_mean': sigma / valid,
                            f'{prefix}/text_tokens_mean': tokens / valid,
                            f'{prefix}/loss_contribution': summed / total})
            if valid > 1:
                # Dispersion of per-example losses, not pixel-level uncertainty
                # or a confidence interval for correlated training observations.
                variance = max(0., (squared - summed * mean) / (valid - 1))
                payload[f'{prefix}/loss_variance'] = variance
                payload[f'{prefix}/loss_std'] = math.sqrt(variance)
        return payload

    def end_update(self, step):
        if not self.enabled:
            return {}
        self.updates += 1
        self.output_updates += int(self.sample_outputs)
        if int(step) % self.interval:
            return {}
        packed = torch.cat((self._base.flatten(), self._output.flatten()))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        host = packed.cpu().tolist()
        columns = self._base.shape[1]
        base = [host[i * columns:(i + 1) * columns] for i in range(BASE_FIELDS)]
        offset = BASE_FIELDS * columns
        outputs = [host[offset + i * BINS:offset + (i + 1) * BINS] for i in range(OUTPUT_FIELDS)]
        total = sum(base[0][:BINS])
        payload = {'diagnostics/window_updates': self.updates,
                   'diagnostics/output_sampled_updates': self.output_updates,
                   'diagnostics/samples': total,
                   'diagnostics/nonfinite_samples': sum(base[0][:BINS]) - sum(base[1][:BINS])}
        for group_id, group in enumerate(self.groups):
            for i in range(BINS):
                column = group_id * BINS + i
                row = [values[column] for values in base]
                prefix = f'diagnostics/{group}/sigma_{i / 10:.1f}-{(i + 1) / 10:.1f}'
                # Publish zeros for empty strata to show their coverage; leave
                # loss absent, never fabricate a zero loss or carry an old EMA.
                payload.update(self._loss_metrics(row, prefix, group == 'all', total))
            if group != 'all':
                row = [sum(values[group_id * BINS:(group_id + 1) * BINS]) for values in base]
                prefix = f'diagnostics/{group}'
                payload.update(self._loss_metrics(row, prefix))
                payload[f'{prefix}/fraction'] = row[0] / total if total else 0.
        for i in range(BINS):
            count, valid, loss, p2, v2, x02, eps2, dot, pmean, vmean, xmean, epsmean = [values[i] for values in outputs]
            prefix = f'diagnostics/output/sigma_{i / 10:.1f}-{(i + 1) / 10:.1f}'
            payload[f'{prefix}/samples'] = count
            payload[f'{prefix}/nonfinite_samples'] = count - valid
            if not valid or valid != count:
                continue
            error_mean = (pmean - vmean) / valid
            payload.update({f'{prefix}/loss_mean': loss / valid,
                            f'{prefix}/error_rms': math.sqrt(max(0., loss / valid)),
                            f'{prefix}/error_mean': error_mean,
                            f'{prefix}/error_variance': max(0., loss / valid - error_mean**2)})
            for name, first, second in (('pred', pmean, p2), ('target', vmean, v2),
                                        ('latent', xmean, x02), ('noise', epsmean, eps2)):
                mean = first / valid
                payload.update({f'{prefix}/{name}_mean': mean,
                                f'{prefix}/{name}_rms': math.sqrt(max(0., second / valid)),
                                f'{prefix}/{name}_variance': max(0., second / valid - mean**2)})
            if v2 > 0:
                payload[f'{prefix}/normalized_mse'] = loss / v2
                payload[f'{prefix}/projection_gain'] = dot / v2
            if p2 > 0 and v2 > 0:
                payload[f'{prefix}/pred_target_cosine'] = dot / math.sqrt(p2 * v2)
        self._base.zero_()
        self._output.zero_()
        self.updates = self.output_updates = 0
        return payload
