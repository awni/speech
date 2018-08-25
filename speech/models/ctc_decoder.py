"""
Author: Awni Hannun

This is an example CTC decoder written in Python. The code is
intended to be a simple example and is not designed to be
especially efficient.

The algorithm is a prefix beam search for a model trained
with the CTC loss function.

For more details checkout either of these references:
  https://distill.pub/2017/ctc/#inference
  https://arxiv.org/abs/1408.2873

"""

import numpy as np
import math
import collections

import speech

from speech.models.lm import LanguageModel

NEG_INF = -float("inf")


def make_new_beam():
    fn = lambda: (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)


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


def decode(probs, lm=None, beam_size=10, space=None, blank=0):
    """
    Performs inference for the given output probabilities.

    Arguments:
      probs: The output probabilities (e.g. post-softmax) for each
        time step. Should be an array of shape (time x output dim).
      lm: A language model scorer.
      beam_size (int): Size of the beam to use during inference.
      space (int): Index of the CTC space label.
      blank (int): Index of the CTC blank label.

    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    T, S = probs.shape
    probs = np.log(probs)

    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank
    # (in log space).
    beam = [(tuple(), (0.0, NEG_INF))]

    for t in range(T):  # Loop over time

        # A default dictionary to store the next step candidates.
        next_beam = make_new_beam()

        for s in range(S):  # Loop over vocab
            p = probs[t, s]

            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, (p_b, p_nb) in beam:  # Loop over beam

                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == blank:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)
                    continue

                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated.
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s != end_t:
                    n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:
                    # We don't include the previous probability of not ending
                    # in blank (p_nb) if s is repeated at the end. The CTC
                    # algorithm merges characters not separated by a blank.
                    n_p_nb = logsumexp(n_p_nb, p_b + p)

                # *NB* this would be a good place to include an LM score.
                if s == space and lm is not None:
                    lm_score = lm.score(list(prefix))
                    n_p_nb = logsumexp(n_p_nb, p_b + p + lm_score, p_nb + p + lm_score)

                next_beam[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if s == end_t:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = logsumexp(n_p_nb, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                      key=lambda x: logsumexp(*x[1]),
                      reverse=True)
        beam = beam[:beam_size]

    best = beam[0]
    return best[0], -logsumexp(*best[1])


if __name__ == "__main__":
    np.random.seed(3)

    time = 50
    output_dim = 20

    probs = np.random.rand(time, output_dim)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate decoder with language model.")
    parser.add_argument("config", help="A json file with the language model configuration.")

    args = parser.parse_args()

    with open(args.config, 'r') as fid:
        config = json.load(fid)

    _, preproc = speech.load(config["save_path"], tag="best")

    lang_cfg = config["lang_model"]

    lm = LanguageModel(path=lang_cfg["path"],
                       alpha=lang_cfg["alpha"],
                       beta=lang_cfg["beta"],
                       preproc=preproc)

    labels, score = decode(probs=probs,
                           lm=lm,
                           space=preproc.space,
                           blank=preproc.vocab_size)
    print labels, score
    print("Score {:.3f}".format(score))
