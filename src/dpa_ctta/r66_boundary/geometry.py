"""Array-only local signed-boundary proposal; no model, loader or optimizer."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates


def crossing(values):
    """One inside-positive/outside-negative crossing on [-1,0,1], else abstain."""
    v = np.asarray(values)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError('finite three-sample affinities required')
    valid = (v[:, 0] > 1e-6) & (v[:, 2] < -1e-6)
    delta = np.zeros(len(v))
    left = valid & (v[:, 1] < 0)
    right = valid & ~left
    delta[left] = -1 + v[left, 0] / (v[left, 0] - v[left, 1])
    delta[right] = v[right, 1] / (v[right, 1] - v[right, 2])
    return delta, valid


def reconstruct(mask, edge, shifts):
    """Nearest-boundary extension gives one deterministic owner to each pixel."""
    if mask.shape not in ((128, 128), (512, 512)) or mask.dtype != bool or edge.shape != mask.shape or edge.dtype != bool:
        raise ValueError('boolean128/512px mask and boundary required')
    scale = mask.shape[0] / 128
    if shifts.shape != mask.shape or not np.isfinite(shifts).all() or np.abs(shifts).max() > scale:
        raise ValueError('finite displacement bounded by one native feature pixel required')
    if not edge.any():
        return mask.copy()
    distance = distance_transform_edt(~mask) - distance_transform_edt(mask)
    distance -= .5 * np.sign(distance)
    nearest = distance_transform_edt(~edge, return_distances=False, return_indices=True)
    offset = shifts[tuple(nearest)]
    return np.where(np.abs(distance) <= scale, distance < offset, mask)


def proposal(features, masks):
    f = np.asarray(features)
    if f.ndim != 3 or f.shape[1:] != (128, 128) or f.shape[0] < 2 or not np.isfinite(f).all():
        raise ValueError('finite Cx128x128 features required')
    if masks.shape != (2, 128, 128) or masks.dtype != bool or (masks[1] & ~masks[0]).any():
        raise ValueError('nested boolean OD/OC masks required')
    f = f / np.maximum(np.linalg.norm(f, axis=0, keepdims=True), 1e-8)
    output = masks.copy(); fields = np.zeros((2, 128, 128)); counts = []
    def sample(array, points, order):
        return np.stack([map_coordinates(a.astype(float), points, order=order,
                         mode='constant', cval=0, prefilter=False) for a in array])
    for ch, mask in enumerate(masks):
        edge = mask & ~binary_erosion(mask, border_value=1)
        yx = np.array(np.nonzero(edge), dtype=float)
        if not edge.any() or mask.all():
            counts.append(dict(boundary=0, accepted=0)); continue
        signed = distance_transform_edt(~mask) - distance_transform_edt(mask)
        normal = np.stack(np.gradient(signed))[:, edge]
        norm = np.linalg.norm(normal, axis=0)
        normal /= np.maximum(norm, 1e-8)
        tangent = np.stack((-normal[1], normal[0]))
        origin = yx + .5 * normal
        eligible = norm > 1e-8
        means = []
        for offsets, side in [((-4, -3, -2), mask), ((2, 3, 4), ~mask if ch == 0 else masks[0] & ~mask)]:
            points = np.stack([origin + n*normal + t*tangent for n in offsets for t in (-1, 0, 1)], axis=2)
            within = ((points >= 0) & (points <= 127)).all(0)
            admitted = within & (sample(side[None], points, 0)[0] > .5)
            number = admitted.sum(1)
            mean = (sample(f, points, 1) * admitted[None]).sum(2) / np.maximum(number, 1)[None]
            length = np.linalg.norm(mean, axis=0)
            eligible &= (number >= 4) & (length > 1e-8)
            means.append(mean / np.maximum(length, 1e-8)[None])
        points = np.stack([origin + t*normal for t in (-1, 0, 1)], axis=2)
        eligible &= ((points >= 0) & (points <= 127)).all((0, 2))
        affinity = (sample(f, points, 1) * (means[0]-means[1])[:, :, None]).sum(0)
        delta, crossing_valid = crossing(affinity)
        accepted = eligible & crossing_valid
        fields[ch][edge] = np.where(accepted, delta, 0)
        output[ch] = reconstruct(mask, edge, fields[ch])
        counts.append(dict(boundary=int(edge.sum()), accepted=int(accepted.sum())))
    # Reject incompatible simultaneous edits rather than inventing class priority.
    violation = output[1] & ~output[0]
    output[:, violation] = masks[:, violation]
    return output, fields, dict(channels=counts, nesting_veto_pixels=int(violation.sum()))


def full_resolution(features, masks):
    if masks.shape != (2, 512, 512) or masks.dtype != bool or (masks[1] & ~masks[0]).any():
        raise ValueError('nested boolean512px masks required')
    native = masks[:, 2::4, 2::4]
    _, fields, audit = proposal(features, native)
    output, veto = apply_fields(masks, fields)
    audit['full_resolution_nesting_veto_pixels'] = veto
    return output, fields, audit


def apply_fields(masks, fields):
    if masks.shape != (2, 512, 512) or masks.dtype != bool or (masks[1] & ~masks[0]).any():
        raise ValueError('nested boolean512px masks required')
    if fields.shape != (2, 128, 128) or not np.isfinite(fields).all() or np.abs(fields).max() > 1:
        raise ValueError('finite native fields bounded by one pixel required')
    native = masks[:, 2::4, 2::4]
    output = masks.copy()
    for ch, mask in enumerate(masks):
        edge = native[ch] & ~binary_erosion(native[ch], border_value=1)
        if not edge.any(): continue
        owner = distance_transform_edt(~edge, return_distances=False, return_indices=True)
        dense = fields[ch][tuple(owner)]
        coordinates = (np.indices((512, 512), dtype=float) + .5) / 4 - .5
        shifted = 4 * map_coordinates(dense, coordinates, order=0, mode='nearest', prefilter=False)
        output[ch] = reconstruct(mask, mask & ~binary_erosion(mask, border_value=1), shifted)
    violation = output[1] & ~output[0]
    output[:, violation] = masks[:, violation]
    return output, int(violation.sum())


def self_check():
    mask = np.zeros((128, 128), bool); mask[:, :64] = True
    masks = np.stack((mask, np.zeros_like(mask)))
    xx = np.indices(mask.shape)[1]
    movements = []
    for displacement in (-.8, .8):
        p = 1/(1+np.exp(np.clip(4*(xx-(63.5+displacement)), -60, 60)))
        out, field, audit = proposal(np.stack((p, 1-p)), masks)
        central = field[0, 8:-8, 63]
        assert np.all(np.sign(central) == np.sign(displacement))
        assert np.sign(int(out[0].sum())-int(mask.sum())) == np.sign(displacement)
        again = proposal(np.stack((p, 1-p)), masks)
        assert np.array_equal(out, again[0]) and np.array_equal(field, again[1])
        movements.append(float(np.median(central)))
    constant = np.ones((2, 128, 128))
    for m in (masks, np.zeros_like(masks), np.ones_like(masks)):
        out, field, audit = proposal(constant, m)
        assert np.array_equal(out, m) and not field.any()
    edge = mask & ~binary_erosion(mask, border_value=1)
    assert np.array_equal(reconstruct(mask, edge, np.zeros(mask.shape)), mask)
    _, valid = crossing(np.array([[1, -1, 1], [0, 0, 0], [-1, 0, 1], [1, 0, -1]]))
    assert valid.tolist() == [False, False, False, True]
    disconnected = np.zeros_like(mask); disconnected[20:35, 20:35] = True; disconnected[80:100, 80:100] = True
    out, _, _ = proposal(constant, np.stack((disconnected, np.zeros_like(mask))))
    assert np.array_equal(out[0], disconnected)
    _, _, audit = proposal(np.stack((xx/127, 1-xx/127)), np.stack((mask, mask)))
    assert audit['channels'][1]['accepted'] == 0  # No disc rim to seed the cup exterior.
    full = np.repeat(np.repeat(masks, 4, 1), 4, 2)
    full[0, 200, 257] = True  # A sub-grid detail must survive zero displacement.
    assert np.array_equal(full_resolution(constant, full)[0], full)
    try: proposal(np.full_like(constant, np.nan), masks)
    except ValueError: pass
    else: raise AssertionError('nonfinite accepted')
    return dict(status='PASS', synthetic_only=True, inward_outward_median=movements,
                deterministic=True, constant_abstention=True, empty_full=True,
                multiple_crossing_abstention=True, disconnected_identity=True,
                zero_shift_identity=True, full_resolution_detail_identity=True,
                missing_rim_abstention=True, real_images_read=0)
