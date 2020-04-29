import torch
import numpy as np
import random
import os
from transformers import BertTokenizer, BertForMaskedLM
from settings import get_eval_decoding_args
from sampling import sample_from_model
from utils import maybe_create_dir, print_and_save_samples


args = get_eval_decoding_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True


exp_path = './TEXT'
file_name = os.path.join(exp_path, args.method)
maybe_create_dir(exp_path)
maybe_create_dir(file_name)


def main():

    device = torch.device('cuda')

    version = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(version, do_lower_case=version.endswith('uncased'))

    model = BertForMaskedLM.from_pretrained(version)
    model.to(device)
    model.eval()

    TEMPERATURES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                    1.2, 1.4, 1.6, 1.8, 2.0, 5.0]
    TOP_K = [10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 500]
    TOP_P = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    if args.method == 'topk':
        PARAMS = TOP_K
    elif args.method == 'topp':
        PARAMS = TOP_P
    else:
        raise ValueError('%s does not match any known method.' % args.method)

    for temp in TEMPERATURES:

        for param in PARAMS:

            if args.method == 'topk':
                kwargs = {'k': param, 't': temp}
            elif args.method == 'topp':
                kwargs = {'p': param, 't': temp}
            else:
                raise ValueError('%s does not match any known method.' % args.method)

            output_gene = '{}_{}_{}.txt'.format(args.method, param, temp)
            output_path = os.path.join(file_name, output_gene)

            with torch.no_grad():
                gene_sents = sample_from_model(args.nsamples,
                                               args.batch_size,
                                               args.max_lengths,
                                               args.max_iter,
                                               args.burn_in,
                                               model,
                                               tokenizer,
                                               args.method,
                                               **kwargs)

                print_and_save_samples(gene_sents, output_path, tokenizer)


if __name__ == '__main__':
    main()
