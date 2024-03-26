import numpy as np

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

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

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

if __name__ == '__main__':
    gg = GameGraph()
    gg.addUnitTypeTable(envs.real_utt)
    gg_str = str(gg.toTurtle())
    with open("game_graph.ttl", "w") as f:
        f.write(gg_str)

    kg = KG("game_graph.ttl")
    walkers = [RandomWalker(4, with_reverse=True)]

    embeddings, literals = RDF2VecTransformer(walkers=walkers, verbose=1).fit_transform(kg, ["http://microrts.com/game/mainGame"])
    print(embeddings)
    print(literals)
