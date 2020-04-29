import torch
import torch.nn.functional as F
from torch.distributions import Categorical


CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'


device = torch.device('cuda')


def initialize_text(batch_size, max_lengths, tokenizer):
    batch = [[CLS] + [MASK] * max_lengths + [SEP] for _ in range(batch_size)]
    indices = [tokenizer.convert_tokens_to_ids(sent) for sent in batch]
    return torch.tensor(indices, device=device)


def generate_one_step(logits, idx, method, stop_burn_in=False, **kwargs):
    """
    logits: [batch_size, seq_len, vocab_size] --> [batch_size, vocab_size]
    """
    logits = logits[:, idx].squeeze(1)
    temp = kwargs['t']
    probs = F.softmax(logits / temp, dim=-1)

    if 'topk' in method and stop_burn_in is True:
        k = kwargs['k']
        indices_to_remove = probs < torch.topk(probs, k)[0][..., -1, None]
        probs[indices_to_remove] = 0
        input_idx = probs.multinomial(1)

    elif 'topp' in method and stop_burn_in is True:
        p = kwargs['p']
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        sorted_samp_probs = sorted_probs.clone()
        sorted_samp_probs[sorted_indices_to_remove] = 0
        sorted_next_idx = sorted_samp_probs.multinomial(1).view(-1, 1)
        input_idx = sorted_indices.gather(1, sorted_next_idx)

    else:
        input_idx = Categorical(logits=logits).sample().unsqueeze(1)

    return input_idx


def sample_from_model(nsamples, batch_size, max_lengths, max_iter, burn_in, model, tokenizer, method, **kwargs):
    all_batches = []

    while sum([x.size(0) for x in all_batches]) < nsamples:
        inputs = initialize_text(batch_size, max_lengths, tokenizer)

        for iteration in range(max_iter):
            logits = model(inputs)[0]
            mask_id = torch.randint(1, max_lengths, (1,))
            stop_burn_in = iteration > burn_in
            indices = generate_one_step(logits, mask_id, method, stop_burn_in, **kwargs)
            inputs[:, mask_id] = indices

        all_batches += [inputs]

    sentences = torch.cat(all_batches, dim=0)

    return sentences
