import spacy
from spacy.tokens import Span, Doc
from spacy.matcher import PhraseMatcher
import re

class NegationDetection:
    name = 'negation_detection'
    
    def __init__(self, nlp, rule_file, matcher_max_length=10, forward_scope=5, backward_scope=5):
        spacy.tokens.Token.set_extension('negation_type_', default=0)
        spacy.tokens.Token.set_extension('negation_type', getter=lambda t: self.nlp.vocab.strings[t._.negation_type_])

        Span.set_extension('negation', default='AFFIRMED')
        Span.set_extension('is_negated', getter=lambda s: s._.negation == 'NEGATED')
        Span.set_extension('is_possible', getter=lambda s: s._.negation == 'POSSIBLE')
        Span.set_extension('negation_source', default=None)
        
        Doc.set_extension('negs', default=[])
        
        self.nlp = nlp
        # self.matcher = PhraseMatcher(self.nlp.vocab, max_length=matcher_max_length)
        #
        # with open(rule_file) as f:
        #     for rule in f.readlines():
        #         parts = rule.strip().split('\t')
        #         pattern, match_id = parts[0], parts[2][1:-1]
        #         self.matcher.add(match_id, None, self.nlp.tokenizer(pattern.lower()))
        #         self.nlp.vocab.strings.add(match_id)


        # POS and PRE trigger have to be handled seperately, as trigger can occur in both sets and otherwise they are overwritten in dict
        self.matcher1 = PhraseMatcher(self.nlp.vocab, max_length=matcher_max_length)
        with open(rule_file) as f:
            for rule in f.readlines():
                parts = rule.strip().split('\t')
                pattern, match_id = parts[0], parts[2][1:-1]
                if re.match("^POS.*", match_id) or re.match("^CONJ$", match_id) or re.match("^PSEU$", match_id):
                    self.matcher1.add(match_id, None, self.nlp.tokenizer(pattern.lower()))
                    self.nlp.vocab.strings.add(match_id)

        self.matcher2 = PhraseMatcher(self.nlp.vocab, max_length=matcher_max_length)
        with open(rule_file) as f:
            for rule in f.readlines():
                parts = rule.strip().split('\t')
                pattern, match_id = parts[0], parts[2][1:-1]
                if re.match("^PRE.*", match_id) or re.match("^CONJ$", match_id) or re.match("^PSEU$", match_id):
                    self.matcher2.add(match_id, None, self.nlp.tokenizer(pattern.lower()))
                    self.nlp.vocab.strings.add(match_id)
                
        self.forward_scope = forward_scope
        self.backward_scope = backward_scope

    def is_negated(self, tokens):
        return any([t._.negation == 'NEGATED' for t in tokens])

    def is_possible(self, tokens):
        return any([t._.negation == 'POSSIBLE' for t in tokens])
    
    @staticmethod
    def filter_matches(matches):
        def overlap(a, b):
            return a[1] <= b[1] <= a[2] or b[1] <= a[1] <= b[2]
        
        negation_terms = []
        matches_sorted = sorted(matches, key=lambda m: m[2] - m[1], reverse=True)
        for match in matches_sorted:
            # not optimal, but hey it works for now
            if any(overlap(term, match) for term in negation_terms):
                continue
            negation_terms.append(match)
        return sorted(negation_terms, key=lambda m: m[1])
    
    def compute_negations(self, negation_terms, doc):
        # For each sentence, find all negation terms: Go to the first negation term in the sentence (Neg1).
        # If Neg1 is a pseudo-negation term, skip to the next negation term in the sentence.
        # If Neg1 is a pre-condition negation term:
        # Define the scope of Neg1 forward based on the value of $1, terminating the scope when you encounter any of the following: * a termination term * another negation or pseudo-negation term * the end of the sentence
        # If Neg1 is a post-condition negation term, define the scope of Neg1 backwards based on the value of $2.
        # Repeat for all negation terms in the sentence.
        # [PREN] - Prenegation rule tag
        # [POST] - Postnegation rule tag
        # [PREP] - Pre possible negation tag
        # [POSP] - Post possible negation tag
        # [PSEU] - Pseudo negation tag
        # [CONJ] - Conjunction tag

        for term in negation_terms:
            rule_tag, start, end = term
            
            if rule_tag == self.nlp.vocab.strings['PSEU']:
                continue
            
            elif rule_tag == self.nlp.vocab.strings['PREN'] or rule_tag == self.nlp.vocab.strings['PREP']:
                scope_tokens = doc[end: end + self.forward_scope]
                found_entity_idx = -1
                for token in scope_tokens:
                    if token.is_punct or token._.negation_type == 'CONJ':
                        break
                    elif token.ent_type_ == "Medical_condition":
                        found_entity_idx = token.i
                        break
                for ent in doc.ents:
                    if (found_entity_idx >= 0) and (ent.start <= found_entity_idx < ent.end) and (ent.label_ == "Medical_condition"):
                        if rule_tag == self.nlp.vocab.strings['PREN']:
                            ent._.negation = 'NEGATED'
                        elif rule_tag == self.nlp.vocab.strings['PREP']:
                            ent._.negation = 'POSSIBLE'
                        ent._.negation_source = Span(doc, start, end)

            elif rule_tag == self.nlp.vocab.strings['POSN'] or rule_tag == self.nlp.vocab.strings['POSP']:

                scope_tokens = reversed(doc[ max(start - self.backward_scope, 0) : start])

                found_entity_idx = -1
                for token in scope_tokens:
                    if token.is_punct or token._.negation_type == 'CONJ':
                        break
                    elif token.ent_type_ == "Medical_condition":
                        found_entity_idx = token.i
                        break
                for ent in doc.ents:
                    if (found_entity_idx >= 0) and (ent.start <= found_entity_idx < ent.end) and (ent.label_ == "Medical_condition"):
                        if rule_tag == self.nlp.vocab.strings['POSN']:
                            ent._.negation = 'NEGATED'
                        elif rule_tag == self.nlp.vocab.strings['POSP']:
                            ent._.negation = 'POSSIBLE'
                        ent._.negation_source = Span(doc, start, end)
    
    def __call__(self, doc):
        doc_low = Doc(self.nlp.vocab, words=[t.lower_ for t in doc], spaces=[t.whitespace_ for t in doc])
        #print ("doc_low:", doc_low)
        # matches = self.matcher(doc_low)
        # print ("matches1:", matches)
        # # filter matches for overlaps (keep longest span)
        # negation_terms = self.filter_matches(matches)
        # print ("negation terms1:", negation_terms)

        #POS trigger
        matches = self.matcher1(doc_low)
        # filter matches for overlaps (keep longest span)
        negation_terms = self.filter_matches(matches)

        #PRE triger
        matches2 = self.matcher2(doc_low)
        # filter matches for overlaps (keep longest span)
        negation_terms2 = self.filter_matches(matches2)

        negation_terms+=negation_terms2

        doc._.negs = [Span(doc, start, end, label=rule_tag) for rule_tag, start, end in negation_terms]

        #print (">>>", doc._.negs)

        for neg_span in doc._.negs:
            for token in neg_span:
                token._.negation_type_ = neg_span.label
                
        self.compute_negations(negation_terms, doc)
        
        return doc
