"""Order-policy migration helpers; no torch or source-filesystem I/O."""
from copy import deepcopy


def saved_tar_order(cursor):
    if not isinstance(cursor, dict):
        return None
    states = cursor.get("rank_cursors", [cursor])
    orders = [state.get("tar_locality_order") for state in states]
    if not any(order is not None for order in orders):
        return None
    if any(order != orders[0] for order in orders):
        raise ValueError("Checkpoint ranks disagree on TAR locality order")
    return deepcopy(orders[0])


def locality_activation(cursor, epoch, offset, epoch_length):
    """Keep every possibly consumed legacy request before the new order.

    Ready-first requests beyond the committed frontier, including future epochs,
    must also retain their original identity. Unconsumed prefetch is replayable.
    """
    boundary = (int(epoch), int(offset))
    states = cursor.get("rank_cursors", [cursor]) if isinstance(cursor, dict) else []
    for state in states:
        if not isinstance(state, dict):
            continue
        for point in [state, *state.get("future_epochs", [])]:
            point_epoch = int(point.get("epoch", 0))
            end = int(point.get("batch_offset", 0))
            for start, stop in point.get("consumed_ranges", []):
                end = max(end, int(stop))
            if not 0 <= end <= epoch_length:
                raise ValueError("TAR locality migration cursor outside epoch")
            boundary = max(boundary, (point_epoch, end))
    if boundary[1] == epoch_length:
        return boundary[0] + 1, 0
    return boundary
