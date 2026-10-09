"""Exchange Adam memory at matched age while keeping the parameter donor fixed."""
from ..r19_model_only.method import clone, equal


def exchange(parameters, optimizer):
    if parameters['steps'] != optimizer['steps'] or parameters['steps'] <= 0:
        raise ValueError('matched nonzero Adam age required')
    if list(parameters['parameters']) != list(optimizer['parameters']):
        raise ValueError('parameter registration mismatch')
    if not equal(parameters['adam']['param_groups'], optimizer['adam']['param_groups']):
        raise ValueError('optimizer parameter-group mismatch')
    if not equal(parameters['buffers'], optimizer['buffers']) or not equal(parameters['grata'], optimizer['grata']):
        raise ValueError('non-Adam persistent state differs')
    a, b = parameters['adam']['state'], optimizer['adam']['state']
    if list(a) != list(b) or len(a) != len(parameters['parameters']):
        raise ValueError('optimizer ownership mismatch')
    for key in a:
        if set(a[key]) != {'step', 'exp_avg', 'exp_avg_sq'} or set(b[key]) != set(a[key]):
            raise ValueError('unexpected Adam state schema')
        for state in (a[key], b[key]):
            if int(state['step']) != parameters['steps']:
                raise ValueError('Adam local step mismatch')
        if any(a[key][k].shape != b[key][k].shape for k in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('Adam tensor shape mismatch')
    result = clone(parameters)
    result['adam'] = clone(optimizer['adam'])
    return result
