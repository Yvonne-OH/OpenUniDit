"""Explicit model dispatch shared by training and inference."""
from collections.abc import Mapping


def mhc_config(cfg):
    raw=cfg.get('mhc',False)
    if isinstance(raw,bool):return {'enabled':raw}
    if not isinstance(raw,Mapping):raise TypeError('mhc must be a boolean or mapping')
    result=dict(raw)
    if not isinstance(result.get('enabled',False),bool):raise TypeError('mhc.enabled must be true or false')
    result.setdefault('enabled',False)
    return result


def build_diffusion_model(cfg):
    from models.giga_transformer import GigaExpTransformer
    model_cfg=dict(cfg['model']);kind=model_cfg.pop('type','giga');hc=mhc_config(cfg)
    if hc['enabled']:
        if kind!='giga':raise ValueError('mHC requires model.type=giga')
        if cfg.get('stage') == 'stage_1_mhc':
            # The original mHC experiments use their own checkpoint layout.
            from models.giga_transformer_mhc import GigaExpTransformerMHC
            return GigaExpTransformerMHC(**model_cfg)
        from models.giga_transformer_mhc_transfer import GigaExpTransformerMHC
        for name,default in [('n_streams',4),('sinkhorn_iters',20),('mhc_mapping_fp32',True),('init_mode','transfer'),('routing_backend','eager')]:
            model_cfg[name]=hc.get(name,model_cfg.get(name,default))
        return GigaExpTransformerMHC(**model_cfg)
    # No implicit activation from model.n_streams; disabled/default configs use the dense model.
    for name in ['n_streams','sinkhorn_iters','mhc_mapping_fp32','init_mode','routing_backend']:
        model_cfg.pop(name,None)
    if kind=='giga':return GigaExpTransformer(**model_cfg)
    if kind=='wan5b_ss':
        from models.wan5b_single_stream import Wan5BSingleStream
        return Wan5BSingleStream(**model_cfg)
    raise ValueError(f'Unknown model.type: {kind}')
