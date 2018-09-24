from spacy.tokens import Doc


class EntityLinking:
    name = 'entity_linking'
    
    def __init__(self, nlp, candidate_retrievers, disambiguator):
        self.nlp = nlp
        
        if not isinstance(candidate_retrievers, list):
            candidate_retrievers = [candidate_retrievers]
        
        self.candidate_retrievers = candidate_retrievers
        self.disambiguator = disambiguator

        Doc.set_extension('ents_linked', default=[])
        
    def __call__(self, doc):
        entity_candidates = []
        for entity in doc.ents:
            for candidate_retriever in self.candidate_retrievers:
                candidates = candidate_retriever.get_candidates(entity)
                if candidates:
                    entity_candidates.append((entity, candidates))
                    break
        
        doc._.ents_linked = self.disambiguator.disambiguate(entity_candidates)
        
        return doc
