import random
import sys


def gen_seq(n):
    seq = []
    p0_sprout = 0.30
    p0_rep = 0.80
    for i in range(n):
        if seq:
            r = random.random()
            p = p0_sprout if seq[i-1] != 0 else p0_rep
            if r <= p:
                seq.append(0)
            else:
                seq.append(random.getrandbits(8))
        else:
            seq.append(random.getrandbits(8))

    return seq


def encode(seq, p=True):

    n = len(seq)
    if p:
        print("···sequence to encode:", seq)
        print("···size [bytes] of sequence:", n)

    bool_seq = [bool(s) for s in seq]
    encoded = []
    while len(bool_seq) > 0:
        if bool_seq[0]:
            encoded.append(seq[0])
            seq = seq[1:]
            bool_seq = bool_seq[1:]
        else:
            encoded.append(0)
            if True in bool_seq:
                i = bool_seq.index(True)
                encoded.append(len(bool_seq[:i]))
                seq = seq[i:]
                bool_seq = bool_seq[i:]
            else:
                encoded.append(len(bool_seq))
                bool_seq = []
                seq = []

    if p:
        print("···encoded sequence:", encoded)
        print("···size [bytes] of encoded sequence:", len(encoded))
        print("···COMPRESSION RATIO:", round(n/len(encoded), 2))

    return encoded


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Please specify parameter n (size of sequence to encode)")
    else:
        n = sys.argv[1]
        try:
            encode(gen_seq(int(n)))
        except Exception as e:
            print(f'Please provide a valid argument! \n{e}')
    exit()
