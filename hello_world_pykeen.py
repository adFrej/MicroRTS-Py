import numpy as np
import pandas as pd

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
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

if __name__ == '__main__':
    gg = GameGraph()
    gg.processUnitTypeTable(envs.real_utt)
    gg_str = str(gg.toTurtle())
    with open("game_graph.ttl", "w") as f:
        f.write(gg_str)

    triples = gg.getTriples()
    df_triples = pd.DataFrame({"subject": [t[0] for t in triples], "predicate": [t[1] for t in triples], "object": [t[2] for t in triples]}, dtype=str)
    df_triples.to_csv("triples.tsv", index=False, header=False)
    triples_factory = TriplesFactory.from_labeled_triples(triples=df_triples[['subject', 'predicate', 'object']].values)

    training = triples_factory
    validation = triples_factory
    testing = triples_factory

    d = training
    id_to_entity = {v: k for k, v in d.entity_to_id.items()}
    id_to_relation = {v: k for k, v in d.relation_to_id.items()}

    result = pipeline(
        model='TransE',
        loss="softplus",
        training=training,
        testing=testing,
        validation=validation,
        model_kwargs=dict(embedding_dim=5),  # Increase the embedding dimension
        optimizer_kwargs=dict(lr=0.1),  # Adjust the learning rate
        training_kwargs=dict(num_epochs=100, use_tqdm_batch=False),  # Increase the number of epochs
    )

    # The trained model is stored in the pipeline result
    model = result.model

    entity_embeddings = model.entity_representations[0](indices=None).detach().numpy()
    relation_embeddings = model.relation_representations[0](indices=None).detach().numpy()
    print(entity_embeddings)
    print(relation_embeddings)
    print(id_to_entity)
    print(id_to_relation)
