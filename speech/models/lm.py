import kenlm
import numpy as np


class LanguageModel(object):
    def __init__(self, alpha, beta, path, preproc):
        self.alpha = alpha
        self.beta = beta
        self.preproc = preproc
        self.lm = kenlm.LanguageModel(path)

    def score(self, prefix):
        sentence = self._decode(prefix)
        full_score = self.lm.score(sentence, eos=False)

        lm_prob = np.power(10, full_score)

        word_count = len(sentence.strip().split(' '))
        score = self.alpha * np.log(lm_prob) + self.beta * np.log(word_count)
        return score

    def _decode(self, prefix):
        return ''.join(self.preproc.decode(prefix))
