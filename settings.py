import argparse


def get_eval_decoding_args():
    parser = argparse.ArgumentParser('Decoding strategy for evaluation')

    parser.add_argument('--max_lengths', type=int, default=50)
    parser.add_argument('--nsamples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--ngrams', type=int, default=5,
                        help='ngrams for computing Bleu and SelfBleu')

    parser.add_argument('--method', type=str, required=True,
                        choices=['topk', 'topp'], default='topk')

    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--burn_in', type=int, default=250)

    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()

    return args
