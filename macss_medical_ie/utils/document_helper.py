def doc_to_brat(doc):
    def get_or_create(item2id, item, prefix):
        if item not in item2id:
            item2id[item] = '{}-{}'.format(prefix, len(item2id) + 1)
        return item2id[item]

    entity2id = {}
    for ent in doc.ents:
        get_or_create(entity2id, ent, prefix='T')

    entities = []
    relations = []

    relation_idx = 1
    for ent in doc.ents:
        entity_id = get_or_create(entity2id, ent, prefix='T')
        entity = (entity_id, ent.label_.upper(), [[ent.start_char, ent.end_char]], ent.text)

        # if an entity is negated, we create a 'negation' entity from the negation source and also add an arc between them
        if ent._.is_negated:
            negation_source = ent._.negation_source

            negation_entity_id = get_or_create(entity2id, negation_source, prefix='T')
            negation_entity = (negation_entity_id, 'NEGATION', [[negation_source.start_char, negation_source.end_char]], negation_source.text)
            entities.append(negation_entity)

            relation = ['R%d' % relation_idx, 'negates', [['', negation_entity_id], ['', entity_id]]]
            relations.append(relation)
            relation_idx += 1
        entities.append(entity)

    for entity_tail, entity_head, relation in doc._.rels:
        entity_tail_id = get_or_create(entity2id, entity_tail, prefix='T')
        entity_head_id = get_or_create(entity2id, entity_head, prefix='T')

        relation = ('R%d' % relation_idx, relation, [['', entity_tail_id], ['', entity_head_id]])

        relations.append(relation)
        relation_idx += 1
    
    return {
            'text': doc.text,
            'entities': entities,
            'relations': relations
        }
