import numpy as np
import pandas as pd

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts.word_2_vec_preprocessing import Word2VecPreprocessing, process_graph_entity

envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=2,
    num_bot_envs=1,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.coacAI for _ in range(1)],
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    autobuild=False,
)

from rts import GameGraph


if __name__ == '__main__':
    gg = GameGraph()
    gg.processUnitTypeTable(envs.real_utt)
    gg_str = str(gg.toTurtle())
    with open("game_graph.ttl", "w") as f:
        f.write(gg_str)

    triples = gg.getTriples()
    df_triples = pd.DataFrame(
        {"subject": [t[0] for t in triples], "predicate": [t[1] for t in triples], "object": [t[2] for t in triples]},
        dtype=str)
    df_triples.to_csv("triples.tsv", index=False, header=False)

    kg = KG("game_graph.ttl")
    walkers = [RandomWalker(max_depth=3, with_reverse=False, md5_bytes=None)]

    embeddings, literals = RDF2VecTransformer(walkers=walkers,
                                              embedder=Word2VecPreprocessing(processor=process_graph_entity),
                                              verbose=1).fit_transform(kg, ["http://microrts.com/game/unit/3"])
    print(embeddings)
    print(literals)
