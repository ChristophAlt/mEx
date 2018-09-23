import spacy
from spacy import Span, Doc
from spacy.matcher import PhraseMatcher


class NegationDetection:
    name = 'negation_detection'
    
    def __init__(self, nlp, rule_file, matcher_max_length=10, forward_scope=5, backward_scope=5):
        spacy.tokens.Token.set_extension('negation_type_', default=0)
        spacy.tokens.Token.set_extension('negation_type', getter=lambda t: self.nlp.vocab.strings[t._.negation_type_])

        Span.set_extension('negation', default='AFFIRMED')
        Span.set_extension('is_negated', getter=lambda s: s._.negation == 'NEGATED')
        Span.set_extension('negation_source', default=None)
        
        Doc.set_extension('negs', default=[])
        
        self.nlp = nlp
        self.matcher = PhraseMatcher(self.nlp.vocab, max_length=matcher_max_length)
        
        with open(rule_file) as f:
            for rule in f.readlines():
                parts = rule.strip().split('\t')
                pattern, match_id = parts[0], parts[2][1:-1]
                self.matcher.add(match_id, None, self.nlp.tokenizer(pattern))
                self.nlp.vocab.strings.add(match_id)
                
        self.forward_scope = forward_scope
        self.backward_scope = backward_scope

    def is_negated(self, tokens):
        return any([t._.negation == 'NEGATED' for t in tokens])
    
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
            
            elif rule_tag == self.nlp.vocab.strings['PREN']:
                scope_tokens = doc[end: end + self.forward_scope]
                found_entity_idx = -1
                for token in scope_tokens:
                    if token.is_punct or token._.negation_type == 'CONJ':
                        break
                    elif token.ent_type_:
                        found_entity_idx = token.i
                        break
                for ent in doc.ents:
                    if (found_entity_idx >= 0) and (ent.start <= found_entity_idx < ent.end):
                        ent._.negation = 'NEGATED'
                        ent._.negation_source = Span(doc, start, end)
            
            elif rule_tag == self.nlp.vocab.strings['POSN']:
                scope_tokens = reversed(doc[start - self.backward_scope: start])
                found_entity_idx = -1
                for token in scope_tokens:
                    if token.is_punct or token._.negation_type == 'CONJ':
                        break
                    elif token.ent_type_:
                        found_entity_idx = token.i
                        break
                for ent in doc.ents:
                    if (found_entity_idx >= 0) and (ent.start <= found_entity_idx < ent.end):
                        ent._.negation = 'NEGATED'
                        ent._.negation_source = Span(doc, start, end)
    
    def __call__(self, doc):
        matches = self.matcher(doc)
        # filter matches for overlaps (keep longest span)
        negation_terms = self.filter_matches(matches)

        doc._.negs = [Span(doc, start, end, label=rule_tag) for rule_tag, start, end in negation_terms]
        
        for neg_span in doc._.negs:
            for token in neg_span:
                token._.negation_type_ = neg_span.label
                
        self.compute_negations(negation_terms, doc)
        
        return doc
