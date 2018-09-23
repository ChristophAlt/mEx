import spacy

from macss_medical_ie.pipeline.named_entity_recognition import NER
from macss_medical_ie.pipeline.negation_detection import NegationDetection
from macss_medical_ie.pipeline.relation_extraction import RelationExtraction


class MedicalIEPipeline:
    def __init__(self, language, ner_model_path, neg_detection_trigger_path, re_model_path):
        self.nlp = spacy.load(language, disable=['tagger', 'ner', 'textcat'])

        named_entity_recognition = NER(self.nlp, model_file=ner_model_path)
        negation_detection = NegationDetection(self.nlp, rule_file=neg_detection_trigger_path)
        relation_extraction = RelationExtraction(self.nlp, model_file=re_model_path)

        self.nlp.add_pipe(named_entity_recognition)
        self.nlp.add_pipe(relation_extraction)
        self.nlp.add_pipe(negation_detection)

    def process_text(self, text):
        return self.nlp(text)
