import sys
import re
import numpy as np
from scipy.sparse import coo_matrix
from flair.data import Dictionary


class DensestSubgraphDisambiguator:
    """
    Selects candidates based its connectivity to other mention's candidates
    """
    def __init__(self, cooccurrences, id_dictionary, min_candidates=5):
        self.min_candidates = min_candidates
        self.cooccurrences = cooccurrences
        self.id_dictionary = id_dictionary
    
    # selects the smallest CUI
    def narrow_down(self, m_set):
        # print ("WSD narrow_down...")
        smallest = -1
        final_cui = ""

        for xcui in m_set:
            tmp_val = int(re.sub("C", "", xcui))

            if smallest == -1:
                smallest = tmp_val
                final_cui = xcui
            else:
                if smallest > tmp_val:
                    smallest = tmp_val
                    final_cui = xcui
        if smallest != -1:
            return [final_cui]
        return m_set

    # selects the maximum -> if there are several ones, select the CUI with the lowest ID
    # if use_select_smallest==0 we return all CUI which are the maximum
    def get_max_select_smallest(self, selection):

        max_val=-1
        for s in selection.keys():
            if selection[s]>max_val:
                max_val=selection[s]

        output=[]

        for s in selection.keys():
            if selection[s] == max_val:
                output.append(s)

        output=self.narrow_down(output)

        return output

    # selects the maximum -> if there are several ones, select the first occurring one
    def get_max(self, selection):

        max_val=-1
        for s in selection.keys():
            if selection[s]>max_val:
                max_val=selection[s]

        output=[]

        for s in selection.keys():
            if selection[s] == max_val:
                #return just one element
                # return
                output.append(s)
                return output

        return output

    def disambiguate(self, entity_candidates):
        """
        :param mentions: list of (mention_string, char_start, char_end, candidates)
        :returns list of (mention_string, char_start, char_end, concept, candidates -> score)"""

        candidates = [candidates for _, candidates in entity_candidates]
        candidate_scores = self._densest_subgraph_scores(candidates)

        result_set=[]
        # TODO: candidates unknown to the disambiguator should be handeled more gracefully
        for (entity, candidates), scores in zip(entity_candidates, candidate_scores):
            if not scores:
                scores = {c: 0. for c in candidates}
            result_set.append((entity, scores))

        return result_set

    def _densest_subgraph_scores(self, mentions):

        """:returns list of sorted lists of candidate-score tuples"""
        # create profile for all candidates of all mentions
        # for each candidate concept for each mention, we keep ...
        # ... count the number of connections to other candidates of other mentions

        mentions = [[self.id_dictionary.get_idx_for_item(candidate) for candidate in candidates] for candidates in mentions]
        mentions = [[candidate for candidate in candidates if candidate != 0] for candidates in mentions]
        
        candidate_connectivity = [[0 for _ in cands] for cands in mentions]
        # ... and keep their references,
        candidate_connections = [[set() for _ in cands] for cands in mentions]
        # ... count the number of connections to other mentions
        mention_connectivity = [[0 for _ in cands] for cands in mentions]
        # ... and keep their references
        mention_connections = [[dict() for _ in cands] for cands in mentions]
        num_cands = [len(m) for m in mentions]
        # Keep track of removed vertices
        removed = set()

        # Now we fill profiles with respective information (basically what is connected to what, and how frequent)
        for i in range(len(mentions)-1):
            cands = mentions[i]
            for j in range(i+1, len(mentions)):
                cands2 = mentions[j]
                if cands is not cands2:
                    for k in range(len(cands)):
                        c = cands[k]
                        for n in range(len(cands2)):
                            c2 = cands2[n]
                            ccs = self.cooccurrences.getrow(c2).indices if c2 != 0 else None

                            if ccs is not None and c in ccs:
                                candidate_connectivity[i][k] += 1
                                candidate_connectivity[j][n] += 1
                                mention_connectivity[i][k] += 1
                                mention_connectivity[j][n] += 1
                                candidate_connections[i][k].add((j,n))
                                candidate_connections[j][n].add((i,k))
                                if j not in mention_connections[i][k]:
                                    mention_connections[i][k][j] = 0
                                if i not in mention_connections[j][n]:
                                    mention_connections[j][n][i] = 0
                                mention_connections[i][k][j] += 1
                                mention_connections[j][n][i] += 1

        # Now we remove iteratively select the most ambiguous mention and remove its worst scored (connected) concept candidate
        # until all mentions have at most a specified minimum number of candidates
        while True:
            # get mention with most candidates (most ambiguous)
            max_ambig_m_i = max(range(len(mentions)), key=lambda i: num_cands[i])
            max_ambig_m = mentions[max_ambig_m_i]
            if num_cands[max_ambig_m_i] <= self.min_candidates:
                break
            num_cands[max_ambig_m_i] -= 1

            # get min connected candidate;
            # connectivity score is product of
            # connectivity to other candidates of other mentions and
            # connectivity to other mentions
            min_connected = min(range(len(max_ambig_m)),
                                key=lambda i:
                                candidate_connectivity[max_ambig_m_i][i] * mention_connectivity[max_ambig_m_i][i])

            # remove candidate by setting connectivity to max int
            candidate_connectivity[max_ambig_m_i][min_connected] = sys.maxsize #sys.maxint #
            mention_connectivity[max_ambig_m_i][min_connected] = 1
            removed.add((max_ambig_m_i, min_connected))
            # reduce connectivity of other connected candidates
            for i,k in candidate_connections[max_ambig_m_i][min_connected]:
                if (i,k) not in removed:
                    candidate_connectivity[i][k] -= 1
                    mention_connections[i][k][max_ambig_m_i] -= 1
                    if mention_connections[i][k][max_ambig_m_i] == 0:
                        mention_connectivity[i][k] -= 1

        # from resulting (filtered) candidates,
        # score candidates as above, normalize scores and select highest scoring as disambiguated concept
        result = list()
        for i in range(len(mentions)):
            cands = mentions[i]
            r = dict()
            summ = 0.0
            for k in range(len(cands)):
                c = self.id_dictionary.get_item_for_index(cands[k])
                if (i, k) not in removed:
                    s = float(candidate_connectivity[i][k] * mention_connectivity[i][k])
                    r[c] = s
                    summ += s
                else:
                    r[c] = 0.0
            for c in r.keys(): #r.iterkeys():
                r[c] /= (max(1, summ * (len(mentions)-1)))
            result.append(r)
        return result
    
    def save(self, pickle_file):
        import pickle
        with open(pickle_file, 'wb') as p_file:
            pickle.dump({
                'cooccurrences': self.cooccurrences,
                'id_dictionary': self.id_dictionary
            }, p_file)
    
    @classmethod
    def load_from_file(cls, pickle_file):
        import pickle
        with open(pickle_file, 'rb') as p_file:
            state = pickle.load(p_file)
            return cls(state['cooccurrences'], state['id_dictionary'])
        
    @classmethod
    def load_from_cooccurrence_file(cls, coocurrence_file, min_occurrence=0):
        id_dictionary = Dictionary(add_unk=True)

        row = []
        col = []

        print ('Loading coocurrences, this might take a while...')
        with open(coocurrence_file, 'r') as cooc_file:
            for line in cooc_file:
                if not line.startswith('#'):
                    n_occurrence, cui1, cui2 = line.strip().split('\t')

                    if int(n_occurrence) > min_occurrence:
                        cui1 = id_dictionary.add_item(cui1)
                        cui2 = id_dictionary.add_item(cui2)
                        
                        row.append(cui1)
                        col.append(cui2)
                        
                        row.append(cui2)
                        col.append(cui1)

        print('n connections:', len(row))

        n_cui = len(id_dictionary)

        print(n_cui)
        data = [1 for _ in range(len(row))]
        cooccurrences = coo_matrix((data, (row, col)), dtype=np.bool, shape=(n_cui, n_cui)).tocsr()

        print ("> finished!")

        return cls(cooccurrences, id_dictionary)
