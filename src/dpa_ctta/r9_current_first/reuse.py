"""Reuse is an explicit identity equality, not a matching score or method nickname."""
from .protocol import digest
KEYS={'checkpoint','algorithm_inventory','preprocessing','stream_rows','seed_rng',
      'source_artifact','score_version','output_policy'}


def reuse(old,new,receipt):
    if set(old)!=KEYS or set(new)!=KEYS or any(not x for x in old.values()) or old!=new:
        raise ValueError('R9 exact reuse identity failed')
    if receipt.get('identity_sha256')!=digest(old) or receipt.get('status')!='COMPLETE' or not receipt.get('seal'):
        raise ValueError('verified original sealed receipt required')
    return dict(schema='R9_REUSE_V1',identity_sha256=digest(new),original_receipt=receipt['seal'],
                new_model_forwards=0,new_backward_calls=0,new_optimizer_steps=0)
