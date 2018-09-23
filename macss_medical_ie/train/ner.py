import fire

from os import listdir
from os.path import isfile, join

from sklearn.model_selection import train_test_split

from flair.models import SequenceTagger
from flair.trainers import SequenceTaggerTrainer
from flair.data import TaggedCorpus, Sentence
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings
from typing import List


def load_corpus(path_to_data, dev_size, test_size, seed, tag_to_biloes=None):
    conll_files = [f for f in listdir(path_to_data) if isfile(join(path_to_data, f)) and f.endswith(".conll")]
    
    sentences: List[Sentence] = []
    for f in conll_files:
        sentences.extend(NLPTaskDataFetcher.read_column_data(
            path_to_column_file=join(path_to_data, f),
            column_name_map={0: 'text', 1: 'pos', 2: 'ner'}))
    
    if tag_to_biloes:
        for sentence in sentences:
            sentence.convert_tag_scheme(tag_type=tag_to_biloes, target_scheme='iob')

    sentences_train_dev, sentences_test = train_test_split(sentences, test_size=test_size,
                                                           random_state=seed, shuffle=True)

    sentences_train, sentences_dev = train_test_split(sentences_train_dev, test_size=dev_size,
                                                      random_state=seed, shuffle=True)

    return TaggedCorpus(sentences_train, sentences_dev, sentences_test)


def train(data_dir, model_dir, hidden_dim=256, word_embeddings='de-fasttext',
          use_crf=True, learning_rate=.1, batch_size=32, max_epochs=50,
          seed=0, dev_size=.1, test_size=.2):

    tag_type = 'ner'

    corpus: TaggedCorpus = load_corpus(data_dir, dev_size, test_size, seed, tag_to_biloes=None)
    print('Corpus:', corpus)

    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print('Size of tag dictionary:', len(tag_dictionary))

    embedding_types: List[TokenEmbeddings] = [

        WordEmbeddings(word_embeddings),

        # comment in this line to use character embeddings
        #CharacterEmbeddings(),

        # comment in these lines to use contextual string embeddings
        # CharLMEmbeddings('news-forward'),
        # CharLMEmbeddings('news-backward'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger: SequenceTagger = SequenceTagger(hidden_size=hidden_dim,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=use_crf)

    trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=False)

    trainer.train(model_dir,
                  learning_rate=learning_rate,
                  mini_batch_size=batch_size,
                  max_epochs=max_epochs)


if __name__ == '__main__':
    fire.Fire(train)
