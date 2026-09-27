"""Competition metric, including singletons and unretrieved positives."""
import numpy as np
import logging
import time


def entity_f05(tp, predicted, truth):
    if truth == 0:
        return float(predicted == 0)
    return 1.25 * tp / (0.25 * truth + predicted)


def entity_scores(labels, scores, groups, truth_counts, threshold):
    n = len(truth_counts)
    selected = scores >= threshold
    predicted = np.bincount(groups[selected], minlength=n)
    tp = np.bincount(groups[selected], weights=labels[selected], minlength=n)
    denom = .25 * truth_counts + predicted
    values = np.divide(1.25 * tp, denom, out=np.zeros(n, dtype=float), where=denom > 0)
    values[(truth_counts == 0) & (predicted == 0)] = 1.0
    return values


def macro_f05(labels, scores, groups, truth_counts, threshold):
    if len(truth_counts) == 0:
        raise ValueError("Cannot evaluate an empty reference set")
    return float(entity_scores(labels, scores, groups, truth_counts, threshold).mean())


def best_grid_f05(labels, scores, groups, truth_counts, thresholds):
    """Compute the whole threshold grid in one pair pass rather than K passes."""
    thresholds=np.asarray(sorted(set(thresholds)),dtype=float)
    width=len(thresholds)+1; n=len(truth_counts)
    if not n or not len(thresholds): raise ValueError('Empty references or threshold grid')
    bucket=np.searchsorted(thresholds,scores,side='right')
    keys=np.asarray(groups,dtype=np.int64)*width+bucket
    counts=np.bincount(keys,minlength=n*width).reshape(n,width)
    hits=np.bincount(keys,weights=labels,minlength=n*width).reshape(n,width)
    del keys,bucket
    predicted=np.cumsum(counts[:,1:][:,::-1],axis=1)[:,::-1]
    tp=np.cumsum(hits[:,1:][:,::-1],axis=1)[:,::-1]
    denom=.25*truth_counts[:,None]+predicted
    values=np.divide(1.25*tp,denom,out=np.zeros_like(denom,dtype=float),where=denom>0)
    values[(truth_counts[:,None]==0)&(predicted==0)]=1.
    return float(values.mean(axis=0).max())


def optimize_threshold(labels, scores, groups, truth_counts):
    """Exact sweep over tied score blocks. Higher threshold wins exact ties."""
    if len(truth_counts) == 0:
        raise ValueError("Cannot optimize on an empty reference set")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction")
    count = len(truth_counts)
    predicted = np.zeros(count, dtype=np.int64)
    tp = np.zeros(count, dtype=np.int64)
    values = (truth_counts == 0).astype(float)
    total = float(values.sum())
    best = total / count
    threshold = float(np.nextafter(float(scores.max()), np.inf)) if len(scores) else 1.0000000000000002
    order = np.argsort(-scores, kind="stable")
    start = 0
    last_log=time.monotonic()
    while start < len(order):
        end = start + 1
        score = scores[order[start]]
        while end < len(order) and scores[order[end]] == score:
            end += 1
        for pos in range(start,end):
            idx=order[pos]; group=groups[idx]
            previous=values[group]
            predicted[group]+=1; tp[group]+=labels[idx]
            values[group]=1.25*tp[group]/(.25*truth_counts[group]+predicted[group])
            total+=values[group]-previous
        candidate = total / count
        if candidate > best + 1e-12:
            best, threshold = candidate, float(score)
        start = end
        if time.monotonic()-last_log>30:
            logging.info('Exact threshold sweep: %s/%s candidate scores (%.1f%%)',start,len(order),100*start/max(len(order),1))
            last_log=time.monotonic()
    return {"threshold": threshold, "macro_f05": best}


def report(labels, scores, groups, truth_counts, countries, threshold):
    values = entity_scores(labels, scores, groups, truth_counts, threshold)
    selected = scores >= threshold
    tp = int(labels[selected].sum())
    truth = int(truth_counts.sum())
    retrieved = np.bincount(groups, weights=labels, minlength=len(truth_counts))
    ceiling = np.array([entity_f05(p, p, t) for p, t in zip(retrieved, truth_counts)])
    countries = np.asarray(countries)
    singleton = truth_counts == 0
    return {
        "macro_f05": float(values.mean()), "threshold": float(threshold),
        "references": len(truth_counts), "candidate_pairs": len(labels),
        "true_pairs": truth, "retrieved_true_pairs": int(retrieved.sum()),
        "blocking_pair_recall": float(retrieved.sum() / truth) if truth else None,
        "blocking_macro_f05_ceiling": float(ceiling.mean()),
        "micro_precision_diagnostic": tp / int(selected.sum()) if selected.any() else None,
        "micro_recall_diagnostic": tp / truth if truth else None,
        "singleton_accuracy": float(values[singleton].mean()) if singleton.any() else None,
        "by_country": {str(c): float(values[countries == c].mean()) for c in sorted(set(countries))}
    }
