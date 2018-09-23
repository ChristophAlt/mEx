import spacy

from macss_medical_ie.macss_medial_ie_settings import LANGUAGE, NER_MODEL_PATH, \
    NEGATION_TRIGGER_PATH, RE_MODEL_PATH
from macss_medical_ie.pipeline.named_entity_recognition import NER
from macss_medical_ie.pipeline.negation_detection import NegationDetection
from macss_medical_ie.pipeline.relation_extraction import RelationExtraction


class MedicalIEPipeline:
    _MEDICAL_IE_PIPELINE = None

    @staticmethod
    def get_pipeline():
        if MedicalIEPipeline._MEDICAL_IE_PIPELINE is None:
            pipeline = spacy.load(LANGUAGE, disable=['tagger', 'ner', 'textcat'])

            named_entity_recognition = NER(pipeline, model_file=NER_MODEL_PATH)
            negation_detection = NegationDetection(pipeline, rule_file=NEGATION_TRIGGER_PATH)
            relation_extraction = RelationExtraction(pipeline, model_file=RE_MODEL_PATH)

            pipeline.add_pipe(named_entity_recognition)
            pipeline.add_pipe(relation_extraction)
            pipeline.add_pipe(negation_detection)

            MedicalIEPipeline._MEDICAL_IE_PIPELINE = pipeline
        
        return MedicalIEPipeline._MEDICAL_IE_PIPELINE

    @staticmethod
    def get_annotated_document(text):
        return MedicalIEPipeline.get_pipeline()(text)
