import fire
import json
import logging

from os import listdir, chmod
from os.path import isfile, join, dirname

from subprocess import run, PIPE
from tempfile import NamedTemporaryFile

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from flair.models import TextClassifier
from flair.trainers import TextClassifierTrainer
from flair.data import TaggedCorpus, Sentence, Token
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, CharacterEmbeddings, RelativeOffsetEmbeddings, DocumentCNNEmbeddings
from flair.file_utils import cached_path
from typing import List


def load_idx2item(path_to_file):
    idx2item = {}
    with open(path_to_file) as f:
        for idx, line in enumerate(f.readlines()):
            item, _ = line.split(' ', 1)
            if item in idx2item:
                raise ValueError("Item '{}' at line {} appears" + 
                    "multiple times in embeddings.".format(item, idx))
            idx2item[idx] = item
    return idx2item


def load_sentences_weird(path_to_file, idx2item, is_test=True, attach_id=False):

    def add_offset_to_sentence(sentence, offsets, tag):
        for token, offset in zip(sentence.tokens, offsets):
            token.add_tag(tag, offset)
    
    def int_list_from_string(s):
        return list(map(int, s.split(' ')))
    
    sentences: List[Sentence] = []
    with open(path_to_file) as f:
        for line in f.readlines():
            id_, label, token_indices, offsets_e1, offsets_e2 = line.rstrip().split(':', 4)
            token_indices, offsets_e1, offsets_e2 = [
                int_list_from_string(s) for s in [token_indices, offsets_e1, offsets_e2]]

            sentence: Sentence = Sentence()
            if not is_test:
                sentence.add_label(label)

            for token_idx in map(int, token_indices):
                token = idx2item[token_idx]
                sentence.add_token(Token(token))
            
            add_offset_to_sentence(sentence, map(int, offsets_e1), tag='offset_e1')
            add_offset_to_sentence(sentence, map(int, offsets_e2), tag='offset_e2')

            if attach_id:
                setattr(sentence, 'id_', id_)

            if len(sentence) > 0:
                sentence._infer_space_after()
                sentences.append(sentence)
    return sentences


def load_corpus_weird(path_to_data, dev_size, seed, train_file='train.txt', test_file='test.txt'):
    idx2item = load_idx2item(join(path_to_data, 'vocabulary/embeddings.csv'))

    sentences_train_dev: List[Sentence] = load_sentences_weird(join(path_to_data, train_file),
                                                               idx2item=idx2item, is_test=False)
    sentences_test: List[Sentence] = load_sentences_weird(join(path_to_data, test_file),
                                                          idx2item=idx2item, is_test=True)

    sentences_train, sentences_dev = train_test_split(sentences_train_dev, test_size=dev_size,
                                                      random_state=seed, shuffle=True)

    return TaggedCorpus(sentences_train, sentences_dev, sentences_test)


def load_sentences_jsonl(path_to_file, is_test=False, attach_id=False):

    def add_offset_to_sentence(sentence, span, tag):
        start, end = span
        for i, token in enumerate(sentence.tokens):
            if i >= end:
                token.add_tag(tag, (i + 1) - end)
            elif i < start:
                token.add_tag(tag, i - start)
            else:
                token.add_tag(tag, 0)

    sentences: List[Sentence] = []
    with open(path_to_file) as f:
        for line in f.readlines():
            example = json.loads(line)
            id_, tokens, entities, label = int(example['id']), example['tokens'], example['entities'], example['label']

            sentence: Sentence = Sentence()
            if not is_test:
                sentence.add_label(label)
            
            for token in tokens:
                sentence.add_token(Token(token))
            
            add_offset_to_sentence(sentence, entities[0], tag='offset_e1')
            add_offset_to_sentence(sentence, entities[1], tag='offset_e2')

            if attach_id:
                setattr(sentence, 'id_', id_)

            if len(sentence) > 0:
                sentence._infer_space_after()
                sentences.append(sentence)
    
    return sentences


def load_semeval_corpus_jsonl(path_to_data, dev_size, seed, train_file='train.jsonl', test_file='test.jsonl'):
    sentences_train_dev: List[Sentence] = load_sentences_jsonl(join(path_to_data, train_file))
    sentences_test: List[Sentence] = load_sentences_jsonl(join(path_to_data, test_file))

    sentences_train, sentences_dev = train_test_split(sentences_train_dev, test_size=dev_size,
                                                      random_state=seed, shuffle=True)

    return TaggedCorpus(sentences_train, sentences_dev, sentences_test)

dataset_loader = {
    'macss': load_corpus_weird,
    'semeval': load_semeval_corpus_jsonl
}



def train(data_dir: str, model_dir: str, dataset_format: str='macss', num_filters: int=150,
          word_embeddings: str='de-fasttext', offset_embedding_dim: int=50, learning_rate: float=.1,
          batch_size: int=32, max_epochs: int=50, dropout: float=.5, use_char_embeddings: bool=False,
          seed: int=0, dev_size: float=.1, test_size: float=.2):

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S')

    logging.info(f'Training config: {locals().items()}')

    if dataset_format not in ['macss', 'semeval']:
        raise ValueError(f"Dataset format '{dataset_format}' not supported.")

    corpus: TaggedCorpus = dataset_loader[dataset_format](data_dir, dev_size, seed)
    label_dictionary = corpus.make_label_dictionary()

    logging.info(f'Corpus: {corpus}')
    corpus.print_statistics()

    logging.info(f'Size of label dictionary: {len(label_dictionary)}')
    logging.info(f'Labels: {label_dictionary.get_items()}')

    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings(word_embeddings),
        RelativeOffsetEmbeddings('offset_e1', max_len=200, embedding_dim=offset_embedding_dim),
        RelativeOffsetEmbeddings('offset_e2', max_len=200, embedding_dim=offset_embedding_dim),
    ]

    if use_char_embeddings:
        embedding_types += CharacterEmbeddings()

    document_embeddings: DocumentCNNEmbeddings = DocumentCNNEmbeddings(embedding_types,
                                                                       num_filters=num_filters,
                                                                       dropout=dropout)

    classifier: TextClassifier = TextClassifier(document_embeddings=document_embeddings,
                                                label_dictionary=label_dictionary,
                                                multi_label=False)

    trainer: TextClassifierTrainer = TextClassifierTrainer(classifier, corpus, label_dictionary)

    trainer.train(model_dir,
                  learning_rate=learning_rate,
                  mini_batch_size=batch_size,
                  max_epochs=max_epochs)


def evaluate(test_file, model_file, dataset_format='macss', semeval_scoring=False):
    if semeval_scoring:
        eval_script = cached_path(
            'https://raw.githubusercontent.com/vzhong/semeval/master/dataset/SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl',
            cache_dir='scripts')
        chmod(eval_script, 0o777)

    classifier: TextClassifier = TextClassifier.load_from_file(model_file)
    #sentences_test: List[Sentence] = load_sentences_jsonl(test_file, attach_id=True)
    idx2item = load_idx2item(join(dirname(test_file), 'vocabulary/embeddings.csv'))

    load_dataset = dataset_loader[dataset_format]

    sentences_test: List[Sentence] = load_dataset(test_file, idx2item, is_test=False, attach_id=True)
    sentences_pred: List[Sentence] = load_dataset(test_file, idx2item, is_test=True, attach_id=True)                                                      

    sentences_pred = classifier.predict(sentences_pred)

    if semeval_scoring:
        id_labels_true = [(sentence.id_, sentence.labels[0]) for sentence in sentences_test]
        id_labels_pred = [(sentence.id_, sentence.labels[0]) for sentence in sentences_pred]

        input_files = []
        for id_labels in [id_labels_true, id_labels_pred]:
            tmp_file = NamedTemporaryFile(delete=True)
            input_files.append(tmp_file)
            with open(tmp_file.name, 'w') as f:
                for id_, label in id_labels:
                    f.write('{}\t{}\n'.format(id_, label.name))
            tmp_file.file.close()

        p = run([eval_script, input_files[0].name, input_files[1].name], stdout=PIPE, encoding='utf-8')
        main_result = p.stdout
        print(main_result)

    else:
        y_true = [sentence.labels[0].name for sentence in sentences_test]
        y_pred = [sentence.labels[0].name for sentence in sentences_pred] 
        print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'evaluate': evaluate
    })
