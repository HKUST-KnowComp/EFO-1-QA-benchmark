
from abc import ABC, abstractclassmethod


class AppFOQEstimator(ABC):

    @abstractclassmethod
    def get_entity_embedding(self, entity_ids: list(int)):
        pass

    @abstractclassmethod
    def get_projection_embedding(self, proj_ids: list(int), emb):
        pass

    @abstractclassmethod
    def get_conjunction_embedding(self, lemb, remb):
        pass

    @abstractclassmethod
    def get_disjunction_embedding(self, lemb, remb):
        pass

    @abstractclassmethod
    def get_difference_embedding(self, lemb, remb):
        pass
