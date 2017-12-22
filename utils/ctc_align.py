
import math
import numpy as np

NEG_INF = -float("inf")

def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max)
                        for a in args))
    return a_max + lsp

def insert_eps(labels):
    labels_with_eps = [0]
    for l in labels:
        labels_with_eps.append(l)
        labels_with_eps.append(0)
    return labels_with_eps

def log_softmax(acts):
    acts -= np.max(acts, axis=1, keepdims=True)
    probs = np.sum(np.exp(acts), axis=1, keepdims=True)
    log_probs = acts - np.log(probs)
    return log_probs

def score(acts, labels):
    """
    Compute the CTC forward variables. Assumes
    the blank label is 0.

    Arguments:
        acts (ndarray): Output probabilites of shape (time, vocab)
        labels (list): List of label integers.
    """
    log_probs = log_softmax(acts)
    labels_with_eps = insert_eps(labels)

    T = acts.shape[0]
    S = len(labels_with_eps)

    # Forward
    grid = np.full((T, S), NEG_INF, dtype=np.float32)
    grid[0, 0] = log_probs[0, labels_with_eps[0]]
    grid[0, 1] = log_probs[0, labels_with_eps[1]]

    for t in range(1, T):

        for s in range(S):
            l = labels_with_eps[s]
            if s == 0:
                score = grid[t-1, s]
            else:
                score = logsumexp(grid[t-1, s], grid[t-1, s-1])

            if l != 0 and s > 1 and l != labels_with_eps[s-2]:
                score = logsumexp(score, grid[t-1, s-2])
            grid[t, s] = score + log_probs[t, l]

    # Backward
    bgrid = np.full((T, S), NEG_INF, dtype=np.float32)
    bgrid[-1, -1] = log_probs[-1, labels_with_eps[-1]]
    bgrid[-1, -2] = log_probs[-1, labels_with_eps[-2]]

    for t in reversed(range(0, T-1)):
        for s in range(S):
            l = labels_with_eps[s]
            if s == S-1:
                score = bgrid[t+1, s]
            else:
                score = logsumexp(bgrid[t+1, s], bgrid[t+1, s+1])

            if l != 0 and s < S-2 and l != labels_with_eps[s+2]:
                score = logsumexp(score, bgrid[t+1, s+2])
            bgrid[t, s] = score + log_probs[t, l]

    return grid + bgrid

def align(acts, labels):
    """
    Align the labels to the input with the given probabilities.
    Assumes the blank label is 0.

    Arguments:
        acts (ndarray): Output activations of shape (time, vocab)
        labels (list): List of label integers.
    """
    log_probs = log_softmax(acts)

    labels_with_eps = insert_eps(labels)

    T = acts.shape[0]
    S = len(labels_with_eps)

    ids = np.zeros((T, S), dtype=np.int32)
    grid = np.full((T, S), NEG_INF, dtype=np.float32)
    grid[0, 0] = log_probs[0, labels_with_eps[0]]
    grid[0, 1] = log_probs[0, labels_with_eps[1]]

    for t in range(1, T):
        for s in range(S):
            l = labels_with_eps[s]
            if grid[t-1, s] > grid[t-1, s-1]:
                max_id = s
                score = grid[t-1, s]
            else:
                max_id = s - 1
                score = grid[t-1, s-1]
            if l != 0 and s > 1 and l != labels_with_eps[s-2] \
              and grid[t-1, s-2] > score:
                max_id = s - 2
                score = grid[t-1, s-2]
            grid[t, s] = score + log_probs[t, l]
            ids[t, s] = max_id

    # chase max path back and build alignment
    alignment = []
    a = S-1 if grid[T-1, S-1] > grid[T-1, S-2] else S-2
    alignment = [a]
    for t in range(T - 1, 0, -1):
        alignment.append(ids[t, alignment[-1]])
    return [labels_with_eps[a] for a in reversed(alignment)]

if __name__ == "__main__":
    T = 10
    V = 4
    L = 4
    acts = np.random.rand(T, V)
    labels = np.random.randint(1, V, size=L)
    alignment = align(acts, labels)
    assert len(alignment) == T, \
            "Alignment is wrong length."
    print labels
    print alignment
    grid = score(acts, labels)
    print grid.shape
