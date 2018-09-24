import spacy

from macss_medical_ie.macss_medial_ie_settings import LANGUAGE, NER_MODEL_PATH, \
    NEGATION_TRIGGER_PATH, RE_MODEL_PATH, DISAMBIGUATOR_PATH, UMLS_SOLR_HOST, UMLS_SOLR_PORT, \
    UMLS_SOLR_CORE, ENTITY_LINKING_EDIT_DISTANCE
from macss_medical_ie.pipeline.named_entity_recognition import NER
from macss_medical_ie.pipeline.negation_detection import NegationDetection
from macss_medical_ie.pipeline.relation_extraction import RelationExtraction
from macss_medical_ie.pipeline.entity_linking.solr_umls_candidate_retriever import SolrUMLSCandidateRetriever
from macss_medical_ie.pipeline.entity_linking.densest_subgraph_disambiguator import DensestSubgraphDisambiguator
from macss_medical_ie.pipeline.entity_linking import EntityLinking


class MedicalIEPipeline:
    _MEDICAL_IE_PIPELINE = None

    @staticmethod
    def get_pipeline():
        if MedicalIEPipeline._MEDICAL_IE_PIPELINE is None:
            pipeline = spacy.load(LANGUAGE, disable=['tagger', 'ner', 'textcat'])

            named_entity_recognition = NER(pipeline, model_file=NER_MODEL_PATH)
            negation_detection = NegationDetection(pipeline, rule_file=NEGATION_TRIGGER_PATH)
            relation_extraction = RelationExtraction(pipeline, model_file=RE_MODEL_PATH)

            disambiguator = DensestSubgraphDisambiguator.load_from_file(DISAMBIGUATOR_PATH)
            entity_linking = EntityLinking(
                pipeline,
                disambiguator=disambiguator,
                candidate_retrievers=[
                    SolrUMLSCandidateRetriever(host=UMLS_SOLR_HOST, port=UMLS_SOLR_PORT,
                        core=UMLS_SOLR_CORE, language='GER',edit_distance=ENTITY_LINKING_EDIT_DISTANCE),
                    SolrUMLSCandidateRetriever(host=UMLS_SOLR_HOST, port=UMLS_SOLR_PORT,
                        core=UMLS_SOLR_CORE, language='ENG', edit_distance=ENTITY_LINKING_EDIT_DISTANCE)
                    ])

            pipeline.add_pipe(named_entity_recognition)
            pipeline.add_pipe(entity_linking)
            pipeline.add_pipe(relation_extraction)
            pipeline.add_pipe(negation_detection)

            MedicalIEPipeline._MEDICAL_IE_PIPELINE = pipeline
        
        return MedicalIEPipeline._MEDICAL_IE_PIPELINE

    @staticmethod
    def get_annotated_document(text):
        return MedicalIEPipeline.get_pipeline()(text)
