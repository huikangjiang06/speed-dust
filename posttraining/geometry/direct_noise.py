"""Additive target-displacement sampling only at the four selected direct sites."""
import math

import torch
from safetensors.torch import load_file

from .calibrate import norm_quantiles
from .common import site_seed
from .gmm import DiagonalGMM


def match_isotropic_norm(effective, epsilon, sigma):
    """Match each token/draw to the original injection norm, before final dtype rounding.

    A zero combined vector has no direction, so it falls back to the original
    epsilon without drawing additional randomness. The return remains normalized
    noise; the caller applies sigma once for injection and its existing reciprocal
    sigma once during reconstruction.
    """
    target=(sigma*epsilon).float().square().sum(-1,keepdim=True).sqrt()
    length=(sigma*effective.float()).square().sum(-1,keepdim=True).sqrt()
    scale=target/length.clamp_min(1e-12)
    return torch.where(length>1e-12,effective.float()*scale,epsilon.float())


class AdditiveGeometryNoise:
    def __init__(self, estimator, bank, root, settings):
        self.estimator,self.bank,self.root,self.settings=estimator,bank,root,settings
        self.mixtures={};self.component_rng={};self.normal_rng={};self.rhos={}
        for site,entry in bank['sites'].items():
            values=load_file(str(root/entry['gmm']['path']))
            self.mixtures[site]=DiagonalGMM(values['weights'],values['means'],values['variances'],estimator.device)
            rho=entry['rho_matched']
            if settings.scale_mode=='intermediate':
                rho=math.sqrt(rho)
            elif settings.scale_mode=='full':
                rho=1.
            if settings.rho_override is not None:
                rho=settings.rho_override
            self.rhos[site]=rho
            self.component_rng[site]=torch.Generator(estimator.device).manual_seed(site_seed(settings.seed,site,'component',estimator.rank))
            self.normal_rng[site]=torch.Generator(estimator.device).manual_seed(site_seed(settings.seed,site,'normal',estimator.rank))
        self.reset_logs()

    def reset_logs(self):
        self.logs={}
        self.last_effective={}

    def _log(self,site):
        if site not in self.logs:
            self.logs[site]=dict(samples=0,rounding_samples=0,component_counts=torch.zeros(len(self.mixtures[site].weights),dtype=torch.int64,device=self.estimator.device),
                norms={key:[] for key in ('isotropic','gmm','combined','raw_combined','target_distance','effective_increment')},
                noise_norm_ratios=[],
                rewards=[],raw_rewards=[],reward_samples=0,nonfinite=0,draws=0)
        return self.logs[site]

    def sample(self,site,shape,clean_output,sigma):
        # Always draw the original epsilon first, preserving its exact generator call order.
        eps=self.estimator.noise(*shape)
        if site not in self.mixtures or self.rhos[site]==0:
            return eps
        mixture=self.mixtures[site]
        target,components=mixture.sample(shape,self.component_rng[site],self.normal_rng[site])
        clean=clean_output.detach().float().unsqueeze(0).expand(shape)
        displacement=target-clean
        effective=eps.float()+(self.rhos[site]/sigma)*displacement
        raw_effective=effective
        if self.settings.norm_matching=='isotropic_per_token':
            effective=match_isotropic_norm(effective,eps,sigma)
        returned=effective.to(eps.dtype)
        valid=self.estimator.mask.bool().unsqueeze(0).expand(shape[:-1])
        record=self._log(site);record['draws']+=shape[0]
        record['component_counts']+=torch.bincount(components[valid],minlength=len(mixture.weights))
        record['nonfinite']+=int((~torch.isfinite(effective)).sum().item())
        remaining=self.settings.log_samples_per_site-record['samples']
        if remaining>0:
            locations=valid.flatten().nonzero().flatten()[:remaining]
            e=eps.reshape(-1,shape[-1])[locations]
            d=displacement.reshape(-1,shape[-1])[locations]
            a=effective.reshape(-1,shape[-1])[locations]
            for key,value in [('isotropic',sigma*e),('gmm',self.rhos[site]*d),('combined',sigma*a),('target_distance',d)]:
                record['norms'][key].append(value.square().sum(-1).sqrt().detach().cpu())
            if self.settings.norm_matching=='isotropic_per_token':
                raw=raw_effective.reshape(-1,shape[-1])[locations]
                injected=(sigma*returned.reshape(-1,shape[-1])[locations]).float()
                original=(sigma*e).float()
                record['norms']['raw_combined'].append((sigma*raw).square().sum(-1).sqrt().detach().cpu())
                ratio=(injected.square().sum(-1).sqrt()/original.square().sum(-1).sqrt().clamp_min(1e-12))
                record['noise_norm_ratios'].append(ratio.detach().cpu())
            record['samples']+=len(locations)
        return returned

    def observe_injection(self,site,live,perturbed):
        if site not in self.mixtures or self.rhos[site]==0:
            return
        record=self._log(site);remaining=self.settings.log_samples_per_site-record['rounding_samples']
        if remaining>0:
            copies=live.shape[0]//self.estimator.batch
            mask=self.estimator.mask.repeat(copies,1).bool().flatten()
            idx=mask.nonzero().flatten()[:remaining]
            delta=(perturbed.reshape(-1,live.shape[-1])[idx].float()-live.reshape(-1,live.shape[-1])[idx].float())
            record['norms']['effective_increment'].append(delta.square().sum(-1).sqrt().detach().cpu())
            record['rounding_samples']+=len(idx)

    def observe_reward(self,site,reward,raw_reward):
        if site not in self.mixtures or self.rhos[site]==0:
            return
        record=self._log(site);remaining=self.settings.log_samples_per_site-record['reward_samples']
        if remaining>0:
            valid=self.estimator.valid.unsqueeze(0).expand_as(reward)
            values=reward[valid][:remaining].detach().float().cpu()
            record['rewards'].append(values);record['reward_samples']+=len(values)
            record['raw_rewards'].append(raw_reward[valid][:remaining].detach().float().cpu())

    def report(self):
        output={}
        for site,record in self.logs.items():
            reward=torch.cat(record['rewards']) if record['rewards'] else torch.zeros(1)
            raw_reward=torch.cat(record['raw_rewards']) if record['raw_rewards'] else torch.zeros(1)
            counts=record['component_counts'].cpu().tolist()
            output[site]=dict(rho=self.rhos[site],sigma=self.estimator.sigma(site),draws_on_rank=record['draws'],
                samples=record['samples'],component_counts=counts,component_frequencies=[x/max(1,sum(counts)) for x in counts],
                learned_component_weights=self.mixtures[site].weights.cpu().tolist(),nonfinite_count=record['nonfinite'],
                norm_quantiles={key:norm_quantiles(torch.cat(values)) for key,values in record['norms'].items() if values},
                centered_reward_quantiles=norm_quantiles(reward),centered_reward_mean=reward.mean().item(),
                raw_reward_quantiles=norm_quantiles(raw_reward),raw_reward_rms=raw_reward.square().mean().sqrt().item(),
                centered_reward_rms=reward.square().mean().sqrt().item())
            if self.settings.norm_matching=='isotropic_per_token':
                output[site]['norm_matching']='isotropic_per_token'
                output[site]['injected_noise_norm_ratio_quantiles']=norm_quantiles(torch.cat(record['noise_norm_ratios']))
        return output

    def rng_state(self):
        return dict(component={s:g.get_state().cpu() for s,g in self.component_rng.items()},
                    normal={s:g.get_state().cpu() for s,g in self.normal_rng.items()})

    def load_rng_state(self,state):
        if set(state['component'])!=set(self.component_rng) or set(state['normal'])!=set(self.normal_rng):
            raise ValueError('Saved GMM RNG sites differ from the geometry bank')
        for site in self.component_rng:
            self.component_rng[site].set_state(state['component'][site].cpu())
            self.normal_rng[site].set_state(state['normal'][site].cpu())
