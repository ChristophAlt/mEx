import json


def doc_to_brat(doc, selected_ents=None, selected_rels=None, enable_negation=True,
                enable_candidate_search=True, enable_wsd=True):
    def get_or_create(item2id, item, prefix):
        if item not in item2id:
            item2id[item] = '{}{}'.format(prefix, len(item2id) + 1)
        return item2id[item]

    entity2id = {}
    for ent in doc.ents:
        get_or_create(entity2id, ent, prefix='T')

    entities = []
    relations = []
    attributes = []
    comments = []

    linked_entities = dict(doc._.ents_linked)

    relation_idx = 1
    attribute_idx = 1
    for ent in doc.ents:
        if selected_ents and ent.label_.upper() not in selected_ents:
            continue
        
        entity_id = get_or_create(entity2id, ent, prefix='T')
        entities.append((entity_id, ent.label_.upper(), [[ent.start_char, ent.end_char]], ent.text))

        # create attributes for negated entities
        if enable_negation and ent._.is_negated:
            attributes.append(('A%d' % attribute_idx, 'Negation', entity_id))
            attribute_idx += 1

        if enable_negation and ent._.is_possible:
            attributes.append(('A%d' % attribute_idx, 'Speculation', entity_id))
            attribute_idx += 1

        if enable_candidate_search and ent in linked_entities:
            cuis = [cui for cui, _ in sorted(linked_entities[ent].items(), key=lambda e: -e[1])]

            attributes.append(('A%d' % attribute_idx, 'UMLSCandidate', entity_id))
            attribute_idx += 1

            # TODO: this naming scheme and format should be refactored
            umls_candidates = [{'CUI': cui} for cui in cuis]

            if enable_wsd and umls_candidates:
                umls_candidates = umls_candidates[0]

            comments.append((entity_id, json.dumps({'String': ent.text, 'UMLS-Candidates': umls_candidates})))


    for entity_tail, entity_head, relation_label in doc._.rels:
        if selected_rels and relation_label.upper() not in selected_rels:
            continue
        
        entity_tail_id = get_or_create(entity2id, entity_tail, prefix='T')
        entity_head_id = get_or_create(entity2id, entity_head, prefix='T')

        relations.append(('R%d' % relation_idx, relation_label.upper(), [['', entity_tail_id], ['', entity_head_id]]))
        relation_idx += 1
    
    return {
            'text': doc.text,
            'entities': entities,
            'relations': relations,
            'comments': comments,
            'attributes': attributes
        }
