import os


def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_and_save_samples(sentences, output_path, tokenizer, max_print=5):
    with open(output_path, 'w', encoding='utf-8') as outf:
        for idx, sentence in enumerate(sentences):
            sent_truncated = tokenizer.convert_ids_to_tokens(sentence.tolist()[1:-1])
            outputs = ' '.join([w for w in sent_truncated])
            outf.write(outputs + '\n')

            if idx < max_print:
                print(outputs)

            if idx == max_print:
                print('\n')
