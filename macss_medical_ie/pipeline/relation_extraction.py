from itertools import combinations

import spacy
from spacy.tokens import Span, Doc
from flair.models import TextClassifier
from flair.data import Sentence, Token


class RelationExtraction:
    name = 'relation_extraction'

    def __init__(self, nlp, model_file):
        self.nlp = nlp
        self.clf = TextClassifier.load_from_file(model_file)
        for label in self.clf.label_dictionary.get_items():
            self.nlp.vocab.strings.add(label)
            #split = tag.split('-')
            # add tags without iob prefix to string store
            #if len(split) == 2:
            #    self.nlp.vocab.strings.add(split[1])
        
        Doc.set_extension('rels', default=[])

    @staticmethod
    def _group_entities_by_sentence(doc):
        sentence_ends = [ent.end for ent in doc.sents]

        entity_groups = []
        entity_group = []
        current_sentence = 0
        for entity in doc.ents:
            if entity.end > sentence_ends[current_sentence]:
                if entity_group:
                    entity_groups.append(entity_group)
                    entity_group = []
                current_sentence += 1

            entity_group.append(entity)

        if entity_group:
            entity_groups.append(entity_group)

        return entity_groups
    
    @staticmethod
    def _prepare_sentence(sent, entity1, entity2):
        def add_offset_to_sentence(sentence, span, tag):
            start, end = span
            for i, token in enumerate(sentence.tokens):
                if i >= end:
                    token.add_tag(tag, (i + 1) - end)
                elif i < start:
                    token.add_tag(tag, i - start)
                else:
                    token.add_tag(tag, 0)

        sentence: Sentence = Sentence()
        for token in sent:
            sentence.add_token(Token(token.text))

        sent_offset = sent.start
        add_offset_to_sentence(sentence, (entity1.start - sent_offset, entity1.end - sent_offset), tag='offset_e1')
        add_offset_to_sentence(sentence, (entity2.start - sent_offset, entity2.end - sent_offset), tag='offset_e2')
        
        return sentence
    
    def __call__(self, doc):
        def swap_entities(relation):
            return relation[-6:-1].lower() == 'e2,e1'
        
        def relation_name(relation):
            return relation[:-7]
        
        def negative_relation(relation):
            return relation.lower().startswith('not_')
            
        entity_groups = self._group_entities_by_sentence(doc)
        
        sentences = []
        entity_combinations = []
        for sentence, entities in zip(doc.sents, entity_groups):
            for entity_left, entity_right in combinations(entities, r=2):
                sentences.append(self._prepare_sentence(sentence, entity_left, entity_right))
                entity_combinations.append((entity_left, entity_right))
            
        pred_sentences = self.clf.predict(sentences)
        
        relations = []
        for sent, (ent1, ent2) in zip(pred_sentences, entity_combinations):
            relation = sent.labels[0].name
            
            if not negative_relation(relation):
                if swap_entities(relation):
                    relations.append((ent2, ent1, relation_name(relation)))
                else:
                    relations.append((ent1, ent2, relation_name(relation)))
        
        doc._.rels = relations

        return doc
