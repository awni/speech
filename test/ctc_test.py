
from speech.models import CTC

def test_argmax_decode():
    pre = [1, 2, 2, 0, 0, 0, 2, 1]
    post = [1, 2, 2, 1]
    assert CTC.max_decode(pre) == post

    pre = [2, 2, 2]
    post = [2]
    assert CTC.max_decode(pre) == post

    pre = [0, 0, 0]
    post = []
    assert CTC.max_decode(pre) == post
