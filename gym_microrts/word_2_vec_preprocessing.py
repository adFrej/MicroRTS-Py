import copy
import re
from typing import List, Callable

from pyrdf2vec.embedders import Word2Vec


class Word2VecPreprocessing(Word2Vec):
    def __init__(self, processor: Callable[[str], str], **kwargs):
        super().__init__(**kwargs)
        self.processor = processor

    def fit(self, walks: List[List['SWalk']], is_update: bool = False) -> 'Embedder':
        walks = copy.deepcopy(walks)
        for i in range(len(walks)):
            for j in range(len(walks[i])):
                walks[i][j] = tuple([self.processor(w) for w in walks[i][j]])
        return super().fit(walks, is_update)

    def transform(self, entities: 'Entities') -> 'Embeddings':
        entities = copy.deepcopy(entities)
        for i in range(len(entities)):
            entities[i] = self.processor(entities[i])
        return super().transform(entities)


def process_graph_entity(entity: str) -> str:
    entity = entity.split("#")[-1]

    if entity.startswith("http://microrts.com/game/") and entity.count("/") > 4:
        entity = entity.replace("http://microrts.com/game/", "")
    else:
        entity = entity.replace("http://microrts.com/", "")
    entity = entity.replace("/", ": ")
    # entity = entity.replace("-", " ")
    entity = re.sub(r"([a-z])([A-Z])", lambda m: f"{m.group(1)} {m.group(2).lower()}", entity)
    return entity
