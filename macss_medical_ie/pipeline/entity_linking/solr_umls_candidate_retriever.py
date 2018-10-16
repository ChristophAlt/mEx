import logging
import re
from abc import ABC, abstractmethod
from itertools import chain

import requests


logger = logging.getLogger(__name__)


class CandidateRetriever(ABC):
    
    @abstractmethod
    def get_candidates(self, entity):
        pass

    
class SolrUMLSCandidateRetriever(CandidateRetriever):
    def __init__(self, host, port, core, language, edit_distance=0, top_k=10, str_len_window=(1, 1)):
        self.query_url = f'http://{host}:{port}/solr/{core}/select'
        self.language = language
        self.edit_distance = edit_distance
        self.top_k = top_k
        self.str_len_window = str_len_window
        
    @staticmethod
    def _preprocess_and_split(s_str):
        s_str=re.sub(r"\("," ", s_str)
        s_str=re.sub(r"\)"," ", s_str)
        s_str=re.sub(r"\?"," ? ", s_str)
        s_str=re.sub(r"°","° ", s_str)

        s_str=re.sub(r"\t", " ", s_str)
        s_str=re.sub(r"  *", " ", s_str)
        s_str=re.sub(r"^ ", "", s_str)
        s_str=re.sub(r" $", "", s_str)

        s_str = re.escape(s_str)

        return s_str.split(r'\\s+')
        
    def _get_field_queries(self, entity):
        entity_str_len = entity.end_char - entity.start_char
        str_len_min = entity_str_len - self.str_len_window[0]
        str_len_max = entity_str_len + self.str_len_window[1]
        
        return [
            f'IX_lang:{self.language}',
            f'IX_word_len:{len(entity)}',
            f'IX_str_len:[{str_len_min} TO {str_len_max}]'
        ]
    
    def _get_phrase_query(self, entity):
        edit_distance = self.edit_distance
        
        if edit_distance == 2 and 5 <= len(entity) <= 8:
            edit_distance = 1
        
        elif len(entity) <= 4:
            edit_distance = 0
            
        operator = ' AND ' if edit_distance <= 0 else f'~{edit_distance} AND '
        
        tokens = [self._preprocess_and_split(token.text) for token in entity]
        
        return operator.join(chain.from_iterable(tokens))
        
    def get_candidates(self, entity):
        params = dict(
            q=self._get_phrase_query(entity),
            fq=self._get_field_queries(entity),
            rows=self.top_k,
            sort='IX_str_len asc, IX_cui_occ desc',
            wt='json'
        )
        
        candidates = []
        try:
            response = requests.get(self.query_url, params=params, timeout=5)
        except requests.exceptions.Timeout:
            logger.error(f'Request to {self.query_url} timed out.')
            return candidates
        except requests.exceptions.ConnectionError:
            logger.error(f'Request to {self.query_url} failed.')
            return candidates

        solr_response = response.json()
        solr_response_header = solr_response['responseHeader']
        
        if solr_response_header['status'] != 0:
            logger.error(f'Internal SOLR error: {solr_response}')
            return candidates

        result_docs = solr_response['response']['docs']
        for doc in result_docs:
            candidates.extend(doc['IX_umlsCode'])
        
        return list(set(candidates))
