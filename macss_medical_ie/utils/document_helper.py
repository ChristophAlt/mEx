import json


def doc_to_brat(doc):
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
        entity_id = get_or_create(entity2id, ent, prefix='T')
        entity = (entity_id, ent.label_.upper(), [[ent.start_char, ent.end_char]], ent.text)
        entities.append(entity)

        # create attributes for negated entities
        if ent._.is_negated:
            attributes.append(('A%d' % attribute_idx, 'Negation', entity_id))
            attribute_idx += 1

        if ent in linked_entities:
            cuis = [cui for cui, _ in sorted(linked_entities[ent].items(), key=lambda e: -e[1])]

            attributes.append(('A%d' % attribute_idx, 'UMLSCandidate', entity_id))
            attribute_idx += 1

            # TODO: this naming scheme and format should be refactored
            umls_candidates = [{'CUI': cui} for cui in cuis]
            comments.append((entity_id, json.dumps({'String': ent.text, 'UMLS-Candidates': umls_candidates})))


    for entity_tail, entity_head, relation_name in doc._.rels:
        entity_tail_id = get_or_create(entity2id, entity_tail, prefix='T')
        entity_head_id = get_or_create(entity2id, entity_head, prefix='T')

        relation = ('R%d' % relation_idx, relation_name, [['', entity_tail_id], ['', entity_head_id]])

        relations.append(relation)
        relation_idx += 1
    
    return {
            'text': doc.text,
            'entities': entities,
            'relations': relations,
            'comments': comments,
            'attributes': attributes
        }
