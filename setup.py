import os, json, yaml
import sentencepiece as spm
from datasets import load_dataset
from run import load_tokenizer



def concat_data(train, valid, test):
    data = []
    for split in (train, valid, test):
        for elem in split:
            data.append(elem['src'])
            data.append(elem['trg'])

    with open('data/concat.txt', 'w') as f:
        f.write('\n'.join(data))




def build_vocab():
    assert os.path.exists(f'configs/vocab.yaml')
    with open('configs/vocab.yaml', 'r') as f:
        vocab_dict = yaml.load(f, Loader=yaml.FullLoader)

    assert os.path.exists(f'data/concat.txt')
    opt = f"--input=data/concat.txt\
            --model_prefix=data/spm\
            --vocab_size={vocab_dict['vocab_size']}\
            --character_coverage={vocab_dict['coverage']}\
            --model_type={vocab_dict['type']}\
            --pad_id={vocab_dict['pad_id']} --pad_piece={vocab_dict['pad_piece']}\
            --unk_id={vocab_dict['unk_id']} --unk_piece={vocab_dict['unk_piece']}\
            --bos_id={vocab_dict['bos_id']} --bos_piece={vocab_dict['bos_piece']}\
            --eos_id={vocab_dict['eos_id']} --eos_piece={vocab_dict['eos_piece']}"

    spm.SentencePieceTrainer.Train(opt)
    os.remove('data/concat.txt')



def tokenize_datasets(train, valid, test, tokenizer):
    tokenized_data = []

    for split in (train, valid, test):
        split_tokenized = []
        
        for elem in split:
            temp_dict = dict()
            
            temp_dict['src'] = tokenizer.EncodeAsIds(elem['src'])
            temp_dict['trg'] = tokenizer.EncodeAsIds(elem['trg'])
            
            split_tokenized.append(temp_dict)
        
        tokenized_data.append(split_tokenized)
    
    return tokenized_data




def save_datasets(train, valid, test):
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}
    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)



def filter_dataset(data, min_len=10, max_len=300, max_diff=50):
    filtered = []
    for elem in data:
        temp_dict = dict()
        src_len, trg_len = len(elem['en']), len(elem['de'])
        max_condition = (src_len <= max_len) & (trg_len <= max_len)
        min_condition = (src_len >= min_len) & (trg_len >= min_len)
        diff_condition = abs(src_len - trg_len) < max_diff

        if max_condition & min_condition & diff_condition:
            temp_dict['src'] = elem['en']
            temp_dict['trg'] = elem['de']
            filtered.append(temp_dict)

    return filtered



def main(downsize=True, sort=True):
    #Download datasets
    train = load_dataset('wmt14', 'de-en', split='train')['translation']
    valid = load_dataset('wmt14', 'de-en', split='validation')['translation']
    test = load_dataset('wmt14', 'de-en', split='test')['translation']

    train = filter_dataset(train)
    valid = filter_dataset(valid)
    test = filter_dataset(test)

    if downsize:
        train = train[::100][:30000]
        valid = valid[::2][:1000]
        test = test[::2][:1000]

    if sort:
        train = sorted(train, key=lambda x: len(x['src']))
        valid = sorted(valid, key=lambda x: len(x['src']))
        test = sorted(test, key=lambda x: len(x['src']))

    #Build Vocab
    concat_data(train, valid, test)
    build_vocab()
    
    #Tokenize and Save Datasets
    tokenizer = load_tokenizer()
    train, valid, test = tokenize_datasets(train, valid, test, tokenizer)
    save_datasets(train, valid, test)



if __name__ == '__main__':
    main()
    assert os.path.exists(f'data/train.json')
    assert os.path.exists(f'data/valid.json')
    assert os.path.exists(f'data/test.json')