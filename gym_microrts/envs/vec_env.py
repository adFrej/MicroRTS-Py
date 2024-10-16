import json
import os
import random
import subprocess
import sys
import warnings
import xml.etree.ElementTree as ET
from itertools import cycle

import gym
import jpype
import jpype.imports
import numpy as np
import pandas as pd
import rdflib
from jpype.imports import registerDomain
from jpype.types import JArray, JInt
from PIL import Image

import gym_microrts

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

from gym_microrts.word_2_vec_preprocessing import Word2VecPreprocessing, process_graph_entity

MICRORTS_CLONE_MESSAGE = """
WARNING: the repository does not include the microrts git submodule.
Executing `git submodule update --init --recursive` to clone it now.
"""

MICRORTS_MAC_OS_RENDER_MESSAGE = """
gym-microrts render is not available on MacOS. See https://github.com/jpype-project/jpype/issues/906

It is however possible to record the videos via `env.render(mode='rgb_array')`. 
See https://github.com/vwxyzjn/gym-microrts/blob/b46c0815efd60ae959b70c14659efb95ef16ffb0/hello_world_record_video.py
as an example.
"""


class MicroRTSGridModeVecEnv:
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 150}
    """
    [[0]x_coordinate*y_coordinate(x*y), [1]a_t(6), [2]p_move(4), [3]p_harvest(4), 
    [4]p_return(4), [5]p_produce_direction(4), [6]p_produce_unit_type(z), 
    [7]x_coordinate*y_coordinate(x*y)]
    Create a baselines VecEnv environment from a gym3 environment.
    :param env: gym3 environment to adapt
    """

    def __init__(
        self,
        num_selfplay_envs,
        num_bot_envs,
        partial_obs=False,
        max_steps=2000,
        render_theme=2,
        frame_skip=0,
        ai2s=[],
        map_paths=["maps/10x10/basesTwoWorkers10x10.xml"],
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0]),
        cycle_maps=[],
        autobuild=False,
        jvm_args=[],
        prior=False,
        rdf2vec=False,
        graph_depth=6,
        graph_walks=None,
        graph_reverse=False,
        graph_vector_length=64,
        prior_advice_freq=10,
        seed=1,
        runs_dir=".",
        graph_map_file="grap_map.json",
        graph_ttl_file="graph.ttl",
        graph_triples_file="triples.tsv",
    ):

        self.num_selfplay_envs = num_selfplay_envs
        self.num_bot_envs = num_bot_envs
        self.num_envs = num_selfplay_envs + num_bot_envs
        assert self.num_bot_envs == len(ai2s), "for each environment, a microrts ai should be provided"
        self.partial_obs = partial_obs
        self.max_steps = max_steps
        self.render_theme = render_theme
        self.frame_skip = frame_skip
        self.ai2s = ai2s
        self.map_paths = map_paths
        if len(map_paths) == 1:
            self.map_paths = [map_paths[0] for _ in range(self.num_envs)]
        else:
            assert (
                len(map_paths) == self.num_envs
            ), "if multiple maps are provided, they should be provided for each environment"
        self.reward_weight = reward_weight

        self.microrts_path = os.path.join(gym_microrts.__path__[0], "microrts")

        # prepare training maps
        self.cycle_maps = list(map(lambda i: os.path.join(self.microrts_path, i), cycle_maps))
        self.next_map = cycle(self.cycle_maps)

        if not os.path.exists(f"{self.microrts_path}/README.md"):
            print(MICRORTS_CLONE_MESSAGE)
            os.system(f"git submodule update --init --recursive")

        if autobuild:
            print(f"removing {self.microrts_path}/microrts.jar...")
            if os.path.exists(f"{self.microrts_path}/microrts.jar"):
                os.remove(f"{self.microrts_path}/microrts.jar")
            print(f"building {self.microrts_path}/microrts.jar...")
            root_dir = os.path.dirname(gym_microrts.__path__[0])
            print(root_dir)
            subprocess.run(["bash", "build.sh", "&>", "build.log"], cwd=f"{root_dir}")

        # read map
        root = ET.parse(os.path.join(self.microrts_path, self.map_paths[0])).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))

        # launch the JVM
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            jars = [
                "microrts.jar",
                "lib/bots/Coac.jar",
                "lib/bots/Droplet.jar",
                "lib/bots/GRojoA3N.jar",
                "lib/bots/Izanagi.jar",
                "lib/bots/MixedBot.jar",
                "lib/bots/TiamatBot.jar",
                "lib/bots/UMSBot.jar",
                "lib/bots/mayariBot.jar",  # "MindSeal.jar"
            ]
            for jar in jars:
                jpype.addClassPath(os.path.join(self.microrts_path, jar))
            jpype.startJVM(*jvm_args, convertStrings=False)

        # start microrts client
        from rts.units import UnitTypeTable

        self.real_utt = UnitTypeTable()
        from ai.reward import (
            AttackRewardFunction,
            ProduceBuildingRewardFunction,
            ProduceCombatUnitRewardFunction,
            ProduceWorkerRewardFunction,
            ResourceGatherRewardFunction,
            RewardFunctionInterface,
            WinLossRewardFunction,
        )

        self.rfs = JArray(RewardFunctionInterface)(
            [
                WinLossRewardFunction(),
                ResourceGatherRewardFunction(),
                ProduceWorkerRewardFunction(),
                ProduceBuildingRewardFunction(),
                AttackRewardFunction(),
                ProduceCombatUnitRewardFunction(),
                # CloserToEnemyBaseRewardFunction(),
            ]
        )
        self.start_client()

        self.prior = prior
        if self.prior:
            self.rdf2vec = rdf2vec
            if not os.path.exists(os.path.join(runs_dir, graph_map_file)):
                from rts import GameGraph
                gg = GameGraph()
                gg.processUnitTypeTable(self.real_utt)
                gg_str = str(gg.toTurtle())
                os.makedirs(runs_dir, exist_ok=True)
                with open(os.path.join(runs_dir, graph_ttl_file), "w") as f:
                    f.write(gg_str)

                triples = gg.getTriples()
                df_triples = pd.DataFrame({"subject": [t[0] for t in triples],
                                           "predicate": [t[1] for t in triples],
                                           "object": [t[2] for t in triples]}, dtype=str)
                df_triples.to_csv(os.path.join(runs_dir, graph_triples_file), index=False, header=False)

                if self.rdf2vec:
                    uts = [str(t) for t in gg.getUnitTypes()]
                    ats = [str(t) for t in gg.getActionTypes()]

                    if graph_reverse:
                        graph_depth //= 2
                        graph_walks = int((graph_walks/(1 + len(uts) + len(ats)))**(1/2)) if graph_walks is not None else None # sth wrong
                    else:
                        graph_walks = graph_walks//(1 + len(uts) + len(ats)) if graph_walks is not None else None

                    kg = KG(os.path.join(runs_dir, graph_ttl_file))
                    walkers = [RandomWalker(
                        max_depth=graph_depth,
                        max_walks=graph_walks,
                        with_reverse=graph_reverse,
                        random_state=seed,
                        n_jobs=6,
                        md5_bytes=None)]

                    embeddings, literals = RDF2VecTransformer(
                        walkers=walkers,
                        embedder=Word2VecPreprocessing(processor=process_graph_entity, vector_size=graph_vector_length, workers=1),
                        verbose=1,
                    ).fit_transform(kg, ["http://microrts.com/game/mainGame"] + uts + ats)
                    uts_map = {int(ut.split('/')[-1]): emb for ut, emb in zip(uts, embeddings[1:len(uts) + 1])}
                    uts_map[-1] = np.zeros(embeddings[0].shape[0])
                    ats_map = {int(at.split('/')[-1]): emb for at, emb in zip(ats, embeddings[len(uts) + 1:])}
                    ats_map[-1] = np.zeros(embeddings[0].shape[0])
                    self.graph_map = {
                        "mainGame": embeddings[0],
                        **{f"unitType_{k}": v for k, v in uts_map.items()},
                        **{f"actionType_{k}": v for k, v in ats_map.items()},
                    }
                    with open(os.path.join(runs_dir, graph_map_file), "w") as f:
                        f.write(json.dumps({k: v.tolist() for k, v in self.graph_map.items()}, indent=4))
                    self.uts_map = uts_map
                    self.ats_map = ats_map
            else:
                if self.rdf2vec:
                    with open(os.path.join(runs_dir, graph_map_file), "r") as f:
                        self.graph_map = {k: np.array(v) for k, v in json.load(f).items()}
                    self.uts_map = {int(k.split("_")[-1]): v for k, v in self.graph_map.items() if k.startswith("unitType_")}
                    self.ats_map = {int(k.split("_")[-1]): v for k, v in self.graph_map.items() if k.startswith("actionType_")}

                with open(os.path.join(runs_dir, graph_ttl_file), "r") as f:
                    gg_str = f.read()

            self.graph = rdflib.Graph()
            self.graph.parse(data=gg_str, format="turtle")
            self.step_obs = 0
            self.advice_freq = prior_advice_freq
            self.advice_cache = np.zeros((self.height * self.width), dtype=np.int32) - 1

        # computed properties
        # [num_planes_hp(5), num_planes_resources(5), num_planes_player(3),
        # num_planes_unit_type(z), num_planes_unit_action(6), num_planes_terrain(2)]

        self.num_planes = [5, 5, 3, len(self.utt["unitTypes"]) + 1, 6, 2]
        if partial_obs:
            self.num_planes = [5, 5, 3, len(self.utt["unitTypes"]) + 1, 6, 2, 2]  # 2 extra for visibility
        obs_length = sum(self.num_planes)
        obs_length += len(self.graph_map["mainGame"]) * 3 if self.prior and self.rdf2vec else 0
        obs_length += 7 if self.prior and not self.rdf2vec else 0
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.height, self.width, obs_length), dtype=np.int32
        )

        self.num_planes_len = len(self.num_planes)
        self.num_planes_prefix_sum = [0]
        for num_plane in self.num_planes:
            self.num_planes_prefix_sum.append(self.num_planes_prefix_sum[-1] + num_plane)

        self.action_space_dims = [6, 4, 4, 4, 4, len(self.utt["unitTypes"]), 7 * 7]
        self.action_space = gym.spaces.MultiDiscrete(np.array([self.action_space_dims] * self.height * self.width).flatten())
        self.action_plane_space = gym.spaces.MultiDiscrete(self.action_space_dims)
        self.source_unit_idxs = np.tile(np.arange(self.height * self.width), (self.num_envs, 1))
        self.source_unit_idxs = self.source_unit_idxs.reshape((self.source_unit_idxs.shape + (1,)))

    def start_client(self):

        from ai.core import AI
        from ts import JNIGridnetVecClient as Client

        self.vec_client = Client(
            self.num_selfplay_envs,
            self.num_bot_envs,
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_paths,
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt,
            self.partial_obs,
        )
        self.render_client = (
            self.vec_client.selfPlayClients[0] if len(self.vec_client.selfPlayClients) > 0 else self.vec_client.clients[0]
        )
        # get the unit type table
        self.utt = json.loads(str(self.render_client.sendUTT()))

    def reset(self):
        responses = self.vec_client.reset([0] * self.num_envs)
        obs = [self._encode_obs(np.array(ro)) for ro in responses.observation]

        return np.array(obs)

    def _encode_obs(self, obs):
        do_advices = self.prior and self.step_obs % self.advice_freq == 0
        self.step_obs += 1

        # unitTypesMatrix is obs[3]
        # actionTypesMatrix is obs[4]
        if do_advices:
            if self.rdf2vec:
                advices = np.zeros(obs[0].shape, dtype=np.int32)
            else:
                advices = np.zeros((obs[0].shape[0], obs[0].shape[1], 7), dtype=np.int32)
            for i in range(obs[0].shape[0]):
                for j in range(obs[0][i].shape[0]):
                    a = self._advise_action(i, j, obs)
                    advices[i, j] = a if a is not None else -1 if self.rdf2vec else np.zeros(7)
            advices = advices.reshape(-1, len(advices[0, 0]))
            if self.rdf2vec:
                advices = self._ids_to_graph(advices, self.ats_map)
            self.advice_cache = advices

        obs = obs.reshape(len(obs), -1).clip(0, np.array([self.num_planes]).T - 1)
        obs_planes = np.zeros((self.height * self.width, self.num_planes_prefix_sum[-1]), dtype=np.int32)
        obs_planes_idx = np.arange(len(obs_planes))
        obs_planes[obs_planes_idx, obs[0]] = 1

        for i in range(1, self.num_planes_len):
            obs_planes[obs_planes_idx, obs[i] + self.num_planes_prefix_sum[i]] = 1

        if self.prior:
            self.advice_cache = np.where(obs[[3]].T > 0, self.advice_cache, np.zeros_like(self.advice_cache))
            if self.rdf2vec:
                obs_ut = self._ids_to_graph(obs[3] - 1, self.uts_map)
                obs_at = self._ids_to_graph(np.where(obs[3] > 0, obs[4], -1), self.ats_map)
                obs_planes = np.concatenate((obs_planes, obs_ut, obs_at, self.advice_cache), axis=1)
            else:
                obs_planes = np.concatenate((obs_planes, self.advice_cache), axis=1)
        return obs_planes.reshape(self.height, self.width, -1)

    @staticmethod
    def _ids_to_graph(obs_ids, id_map):
        u, inv = np.unique(obs_ids, return_inverse=True)
        return np.array([id_map[x] for x in u])[inv]

    def _advise_action(self, i, j, obs):
        obs_ij = obs[:, i, j]
        unit = obs_ij[3]
        unit = unit - 1
        if unit == -1:
            return None

        neighbours, directions = self._get_neighbours(obs, i, j)
        neighbours_relation = self._relation(obs_ij, neighbours) if len(neighbours) > 0 else np.array([])

        action_nodes = list(self.graph.objects(rdflib.URIRef(f"http://microrts.com/game/unit/{unit}"), rdflib.URIRef("http://microrts.com/game/unit/does")))
        if len(action_nodes) == 0:
            return None

        ratings = {}
        actions = []
        for action_node in action_nodes:
            prefers_self, prefers_target, prefers_friendly, prefers_enemy = self._get_prefers(action_node)
            action_id = self._node_to_id(action_node)
            rating = 0.

            produce_type = None
            if action_id == 4:#creates
                produces = self.graph.objects(rdflib.URIRef(f"http://microrts.com/game/unit/{unit}"), rdflib.URIRef("http://microrts.com/game/unit/produces"))
                action_ratings = self.graph.objects(action, rdflib.URIRef("http://microrts.com/game/action/describedByInCreates"))
                max_produce_rating = float("-inf")
                for p in produces:
                    unit_ratings = int(self.graph.value(subject=p, predicate=rdflib.URIRef("http://microrts.com/game/unit/ranks")))
                    ratings = list(set(action_ratings) & set(unit_ratings))
                    r = 0.
                    for rating in ratings:
                        r += 1. if str(rating).endswith("Good") else 0. if str(rating).endswith("Medium") else -1.
                    if r > max_produce_rating:
                        produce_type = self._node_to_id(p)
                        max_produce_rating = r
                rating += max_produce_rating

            targets_distance = str(self.graph.value(subject=action_node, predicate=rdflib.URIRef("http://microrts.com/game/action/targetsDistance")))
            targets_player = str(self.graph.value(subject=action_node, predicate=rdflib.URIRef("http://microrts.com/game/action/targetsPlayer")))
            if targets_distance == "distant":
                range_ = int(self.graph.value(subject=rdflib.URIRef(f"http://microrts.com/game/unit/{unit}"), predicate=rdflib.URIRef("http://microrts.com/game/unit/hasAttackRange")))
                targets, relative_pos = self._get_distant_targets(obs, i, j, range_, targets_player)
                for t, r in zip(targets, relative_pos):
                    action = self._id_to_action(action_id, r, produce_type)
                    actions.append(action)
                    ratings[action.tobytes()] = rating + self._rate_action_complete(action_node, unit, obs_ij, neighbours, neighbours_relation, prefers_self, prefers_friendly, prefers_enemy, prefers_target, t)

            elif targets_distance == "self":
                action = self._id_to_action(action_id, None, produce_type)
                actions.append(action)
                ratings[action.tobytes()] = rating + self._rate_action_complete(action_node, unit, obs_ij, neighbours, neighbours_relation, prefers_self, prefers_friendly, prefers_enemy)

            elif targets_distance == "adjacent" and len(neighbours) > 0:
                targeted_neighbours = neighbours[neighbours_relation == targets_player]
                targeted_directions = directions[neighbours_relation == targets_player]
                if len(targeted_neighbours) == 0:
                    continue

                if targets_player == "friendly" or targets_player == "enemy":
                    targets_units = [self._node_to_id(u) for u in self.graph.objects(action_node, rdflib.URIRef("http://microrts.com/game/action/targets"))]
                    targeted_directions = targeted_directions[np.isin(targeted_neighbours[:, 3] - 1, targets_units)]
                    targeted_neighbours = targeted_neighbours[np.isin(targeted_neighbours[:, 3] - 1, targets_units)]
                elif targets_player == "empty":
                    targeted_neighbours = [targeted_neighbours[0]]
                    targeted_directions = [random.choice(targeted_directions)]

                for n, d in zip(targeted_neighbours, targeted_directions):
                    action = self._id_to_action(action_id, d, produce_type)
                    actions.append(action)
                    ratings[action.tobytes()] = rating + self._rate_action_complete(action_node, unit, obs_ij, neighbours, neighbours_relation, prefers_self, prefers_friendly, prefers_enemy, prefers_target, n)

        if len(actions) == 0:
            raise ValueError("No actions available")

        return max(actions, key=lambda a: ratings[a.tobytes()])

    @staticmethod
    def _node_to_id(node):
        return int(node.split("/")[-1])

    @staticmethod
    def _id_to_action(action_id, param, produce_type):
        action = np.zeros(7)
        action[0] = action_id

        if param is not None:
            pos = action_id
            if action_id == 5:
                pos += 1
            action[pos] = param

        if action_id == 4:
            action[5] = produce_type

        return action

    def _get_neighbours(self, obs, i, j):
        if obs[3, i, j] == 0:
            return np.array([])
        neighbours = []
        directions = []
        for idx, (x, y) in enumerate([(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1)]):
            if 0 <= x < self.height and 0 <= y < self.width and (obs[3, x, y] > 0 or obs[5, x, y] == 0):
                neighbours.append(obs[:, x, y])
                directions.append(idx)
        return np.array(neighbours), np.array(directions)

    def _get_distant_targets(self, obs, i, j, range_, relation):
        if obs[3, i, j] == 0:
            return [], []
        targets = []
        relative_pos = []
        idx = 0
        for x in range(-range_, range_ + 1):
            for y in range(-range_, range_ + 1):
                if not (x == 0 and y == 0) and 0 <= i + x < self.height and 0 <= j + y < self.width and self._relation(obs[:, i, j], np.array([obs[:, i + x, j + y]])) == relation:
                    targets.append(obs[:, i + x, j + y])
                    relative_pos.append(idx)
                    idx += 1
        return targets, relative_pos

    @staticmethod
    def _relation(obs_ij, obs_other):
        return np.where(obs_other[:, 3] > 0,
                        np.where(obs_other[:, 2] == 0, "neutral", np.where(obs_other[:, 2] == obs_ij[2], "friendly", "enemy")),
                        np.where(obs_other[:, 5] == 0, "empty", "obstacle")
                        )

    def _get_prefers(self, action):
        prefers = self.graph.objects(action, rdflib.URIRef("http://microrts.com/game/action/prefers"))
        prefers_self = []
        prefers_target = []
        prefers_friendly = []
        prefers_enemy = []
        for prefer in prefers:
            in_ = str(self.graph.value(subject=prefer, predicate=rdflib.URIRef("http://microrts.com/game/action/in")))
            if in_ == "self":
                prefers_self.append(prefer)
            elif in_ == "target":
                prefers_target.append(prefer)
            elif in_ == "friendly":
                prefers_friendly.append(prefer)
            elif in_ == "enemy":
                prefers_enemy.append(prefer)
        return prefers_self, prefers_target, prefers_friendly, prefers_enemy

    def _hp_status(self, obs_ij):
        unit = obs_ij[3] - 1
        hp = obs_ij[0]
        max_hp = int(self.graph.value(subject=rdflib.URIRef(f"http://microrts.com/game/unit/{unit}"), predicate=rdflib.URIRef("http://microrts.com/game/unit/hasHp")))
        return "high" if hp > max_hp / 2 else "low"

    @staticmethod
    def _resources_status(obs_ij):
        resources = obs_ij[1]
        return "high" if resources > 0 else "low"

    def _rate_prefers(self, obs_ij, prefers):
        statistic = str(self.graph.value(subject=prefers, predicate=rdflib.URIRef("http://microrts.com/game/action/statistic")))
        value = str(self.graph.value(subject=prefers, predicate=rdflib.URIRef("http://microrts.com/game/action/value")))
        if statistic == "hp":
            return 1. if value == self._hp_status(obs_ij) else -1.
        if statistic == "resources":
            return 1. if value == self._resources_status(obs_ij) else -1.
        if statistic == "action":
            return 1. if obs_ij[4] - 1 == self._node_to_id(value) else 0.
        if statistic == "unit":
            return 1. if obs_ij[3] - 1 == self._node_to_id(value) else 0.
        return None

    def _rate_prefers_many(self, prefers, related_neighbours):
        r = 0.
        for n in related_neighbours:
            for p in prefers:
                r += self._rate_prefers(n, p)
        return r

    def _rate_action(self, action, unit):
        r = 0.
        action_ratings = self.graph.objects(action, rdflib.URIRef("http://microrts.com/game/action/describedBy"))
        unit_ratings = self.graph.objects(rdflib.URIRef(f"http://microrts.com/game/unit/{unit}"), rdflib.URIRef("http://microrts.com/game/unit/ranks"))
        ratings = list(set(action_ratings) & set(unit_ratings))
        if len(ratings) == 0:
            return r
        for rating in ratings:
            r += 1. if str(rating).endswith("Good") else 0. if str(rating).endswith("Medium") else -1.
        return r

    def _rate_action_complete(self, action_node, unit, obs_ij, neighbours, neighbours_relation, prefers_self, prefers_friendly, prefers_enemy, prefers_target=None, target=None):
        r = self._rate_action(action_node, unit)
        r += self._rate_prefers_many(prefers_self, [obs_ij])
        if len(neighbours) > 0:
            r += self._rate_prefers_many(prefers_friendly, neighbours[neighbours_relation == "friendly"])
            r += self._rate_prefers_many(prefers_enemy, neighbours[neighbours_relation == "enemy"])
        if target is not None:
            r += self._rate_prefers_many(prefers_target, [target])
        return r

    def step_async(self, actions):
        actions = actions.reshape((self.num_envs, self.width * self.height, -1))
        actions = np.concatenate((self.source_unit_idxs, actions), 2)  # specify source unit
        # valid actions
        actions = actions[np.where(self.source_unit_mask == 1)]
        action_counts_per_env = self.source_unit_mask.sum(1)
        java_actions = [None] * len(action_counts_per_env)
        action_idx = 0
        for outer_idx, action_count in enumerate(action_counts_per_env):
            java_valid_action = [None] * action_count
            for idx in range(action_count):
                java_valid_action[idx] = JArray(JInt)(actions[action_idx])
                action_idx += 1
            java_actions[outer_idx] = JArray(JArray(JInt))(java_valid_action)
        self.actions = JArray(JArray(JArray(JInt)))(java_actions)

    def step_wait(self):
        responses = self.vec_client.gameStep(self.actions, [0] * self.num_envs)
        reward, done = np.array(responses.reward), np.array(responses.done)
        obs = [self._encode_obs(np.array(ro)) for ro in responses.observation]
        infos = [{"raw_rewards": item} for item in reward]
        # check if it is in evaluation, if not, then change maps
        if len(self.cycle_maps) > 0:
            # check if an environment is done, if done, reset the client, and replace the observation
            for done_idx, d in enumerate(done[:, 0]):
                # bot envs settings
                if done_idx < self.num_bot_envs:
                    if d:
                        self.vec_client.clients[done_idx].mapPath = next(self.next_map)
                        response = self.vec_client.clients[done_idx].reset(0)
                        obs[done_idx] = self._encode_obs(np.array(response.observation))
                # selfplay envs settings
                else:
                    if d and done_idx % 2 == 0:
                        done_idx -= self.num_bot_envs  # recalibrate the index
                        self.vec_client.selfPlayClients[done_idx // 2].mapPath = next(self.next_map)
                        self.vec_client.selfPlayClients[done_idx // 2].reset()
                        p0_response = self.vec_client.selfPlayClients[done_idx // 2].getResponse(0)
                        p1_response = self.vec_client.selfPlayClients[done_idx // 2].getResponse(1)
                        obs[done_idx] = self._encode_obs(np.array(p0_response.observation))
                        obs[done_idx + 1] = self._encode_obs(np.array(p1_response.observation))
        return np.array(obs), reward @ self.reward_weight, done[:, 0], infos

    def step(self, ac):
        self.step_async(ac)
        return self.step_wait()

    def getattr_depth_check(self, name, already_found):
        """
        Check if an attribute reference is being hidden in a recursive call to __getattr__
        :param name: (str) name of attribute to check for
        :param already_found: (bool) whether this attribute has already been found in a wrapper
        :return: (str or None) name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return "{0}.{1}".format(type(self).__module__, type(self).__name__)
        else:
            return None

    def render(self, mode="human"):
        if mode == "human":
            self.render_client.render(False)
            # give warning on macos because the render is not available
            if sys.platform == "darwin":
                warnings.warn(MICRORTS_MAC_OS_RENDER_MESSAGE)
        elif mode == "rgb_array":
            bytes_array = np.array(self.render_client.render(True))
            image = Image.frombytes("RGB", (640, 640), bytes_array)
            return np.array(image)[:, :, ::-1]

    def close(self):
        if jpype._jpype.isStarted():
            self.vec_client.close()
            jpype.shutdownJVM()

    def get_action_mask(self):
        """
        :return: Mask for action types and action parameters,
        of shape [num_envs, map height * width, action types + params]
        """
        # action_mask shape: [num_envs, map height, map width, 1 + action types + params]
        action_mask = np.array(self.vec_client.getMasks(0))
        # self.source_unit_mask shape: [num_envs, map height * map width * 1]
        self.source_unit_mask = action_mask[:, :, :, 0].reshape(self.num_envs, -1)
        action_type_and_parameter_mask = action_mask[:, :, :, 1:].reshape(self.num_envs, self.height * self.width, -1)
        return action_type_and_parameter_mask


class MicroRTSBotVecEnv(MicroRTSGridModeVecEnv):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 150}

    def __init__(
        self,
        ai1s=[],
        ai2s=[],
        partial_obs=False,
        max_steps=2000,
        render_theme=2,
        map_paths="maps/10x10/basesTwoWorkers10x10.xml",
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0]),
        autobuild=True,
        jvm_args=[],
    ):

        self.ai1s = ai1s
        self.ai2s = ai2s
        assert len(ai1s) == len(ai2s), "for each environment, a microrts ai should be provided"
        self.num_envs = len(ai1s)
        self.partial_obs = partial_obs
        self.max_steps = max_steps
        self.render_theme = render_theme
        self.map_paths = map_paths
        self.reward_weight = reward_weight

        # read map
        self.microrts_path = os.path.join(gym_microrts.__path__[0], "microrts")
        if not os.path.exists(f"{self.microrts_path}/README.md"):
            print(MICRORTS_CLONE_MESSAGE)
            os.system(f"git submodule update --init --recursive")

        if autobuild:
            print(f"removing {self.microrts_path}/microrts.jar...")
            if os.path.exists(f"{self.microrts_path}/microrts.jar"):
                os.remove(f"{self.microrts_path}/microrts.jar")
            print(f"building {self.microrts_path}/microrts.jar...")
            root_dir = os.path.dirname(gym_microrts.__path__[0])
            print(root_dir)
            subprocess.run(["bash", "build.sh", "&>", "build.log"], cwd=f"{root_dir}")

        root = ET.parse(os.path.join(self.microrts_path, self.map_paths[0])).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))

        # launch the JVM
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            registerDomain("rts")
            jars = [
                "microrts.jar",
                "lib/bots/Coac.jar",
                "lib/bots/Droplet.jar",
                "lib/bots/GRojoA3N.jar",
                "lib/bots/Izanagi.jar",
                "lib/bots/MixedBot.jar",
                "lib/bots/TiamatBot.jar",
                "lib/bots/UMSBot.jar",
                "lib/bots/mayariBot.jar",  # "MindSeal.jar"
            ]
            for jar in jars:
                jpype.addClassPath(os.path.join(self.microrts_path, jar))
            jpype.startJVM(*jvm_args, convertStrings=False)

        # start microrts client
        from rts.units import UnitTypeTable

        self.real_utt = UnitTypeTable()
        from ai.reward import (
            AttackRewardFunction,
            ProduceBuildingRewardFunction,
            ProduceCombatUnitRewardFunction,
            ProduceWorkerRewardFunction,
            ResourceGatherRewardFunction,
            RewardFunctionInterface,
            WinLossRewardFunction,
        )

        self.rfs = JArray(RewardFunctionInterface)(
            [
                WinLossRewardFunction(),
                ResourceGatherRewardFunction(),
                ProduceWorkerRewardFunction(),
                ProduceBuildingRewardFunction(),
                AttackRewardFunction(),
                ProduceCombatUnitRewardFunction(),
                # CloserToEnemyBaseRewardFunction(),
            ]
        )
        self.start_client()

        # computed properties
        # [num_planes_hp(5), num_planes_resources(5), num_planes_player(5),
        # num_planes_unit_type(z), num_planes_unit_action(6), num_planes_terrain(2)]

        self.num_planes = [5, 5, 3, len(self.utt["unitTypes"]) + 1, 6, 2]
        if partial_obs:
            self.num_planes = [5, 5, 3, len(self.utt["unitTypes"]) + 1, 6, 2, 2]  # 2 extra for visibility
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

    def start_client(self):

        from ai.core import AI
        from ts import JNIGridnetVecClient as Client

        self.vec_client = Client(
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_paths,
            JArray(AI)([ai1(self.real_utt) for ai1 in self.ai1s]),
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt,
            self.partial_obs,
        )
        self.render_client = self.vec_client.botClients[0]
        # get the unit type table
        self.utt = json.loads(str(self.render_client.sendUTT()))

    def reset(self):
        responses = self.vec_client.reset([0 for _ in range(self.num_envs)])
        raw_obs, reward, done, info = np.ones((self.num_envs, 2)), np.array(responses.reward), np.array(responses.done), {}
        return raw_obs

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        responses = self.vec_client.gameStep(self.actions, [0 for _ in range(self.num_envs)])
        raw_obs, reward, done = np.ones((self.num_envs, 2)), np.array(responses.reward), np.array(responses.done)
        infos = [{"raw_rewards": item} for item in reward]
        return raw_obs, reward @ self.reward_weight, done[:, 0], infos


class MicroRTSGridModeSharedMemVecEnv(MicroRTSGridModeVecEnv):
    """
    Similar function to `MicroRTSGridModeVecEnv` but uses shared mem buffers for
    zero-copy data exchange between NumPy and JVM runtimes. Drastically improves
    performance of the environment with some limitations introduced to the API.
    Notably, all games should be performed on the same map.
    """

    def __init__(
        self,
        num_selfplay_envs,
        num_bot_envs,
        partial_obs=False,
        max_steps=2000,
        render_theme=2,
        frame_skip=0,
        ai2s=[],
        map_paths=["maps/10x10/basesTwoWorkers10x10.xml"],
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0]),
        cycle_maps=[],
    ):
        if len(map_paths) > 1 and len(set(map_paths)) > 1:
            raise ValueError("Mem shared environment requires all games to be played on the same map.")

        super(MicroRTSGridModeSharedMemVecEnv, self).__init__(
            num_selfplay_envs,
            num_bot_envs,
            partial_obs,
            max_steps,
            render_theme,
            frame_skip,
            ai2s,
            map_paths,
            reward_weight,
            cycle_maps,
        )

    def _allocate_shared_buffer(self, nbytes):
        from java.nio import ByteOrder
        from jpype.nio import convertToDirectBuffer

        c_buffer = bytearray(nbytes)
        jvm_buffer = convertToDirectBuffer(c_buffer).order(ByteOrder.nativeOrder()).asIntBuffer()
        np_buffer = np.asarray(jvm_buffer, order="C")
        return jvm_buffer, np_buffer

    def start_client(self):

        from ai.core import AI
        from rts import GameState
        from ts import JNIGridnetSharedMemVecClient as Client

        self.num_feature_planes = GameState.numFeaturePlanes
        num_unit_types = len(self.real_utt.getUnitTypes())
        self.action_space_dims = [6, 4, 4, 4, 4, num_unit_types, (self.real_utt.getMaxAttackRange() * 2 + 1) ** 2]
        self.masks_dim = sum(self.action_space_dims)
        self.action_dim = len(self.action_space_dims)

        # pre-allocate shared buffers with JVM
        obs_nbytes = self.num_envs * self.height * self.width * self.num_feature_planes * 4
        obs_jvm_buffer, obs_np_buffer = self._allocate_shared_buffer(obs_nbytes)
        self.obs = obs_np_buffer.reshape((self.num_envs, self.height, self.width, self.num_feature_planes))

        action_mask_nbytes = self.num_envs * self.height * self.width * self.masks_dim * 4
        action_mask_jvm_buffer, action_mask_np_buffer = self._allocate_shared_buffer(action_mask_nbytes)
        self.action_mask = action_mask_np_buffer.reshape((self.num_envs, self.height * self.width, self.masks_dim))

        action_nbytes = self.num_envs * self.width * self.height * self.action_dim * 4
        action_jvm_buffer, action_np_buffer = self._allocate_shared_buffer(action_nbytes)
        self.actions = action_np_buffer.reshape((self.num_envs, self.height * self.width, self.action_dim))

        self.vec_client = Client(
            self.num_selfplay_envs,
            self.num_bot_envs,
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_paths[0],
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt,
            self.partial_obs,
            obs_jvm_buffer,
            action_mask_jvm_buffer,
            action_jvm_buffer,
            0,
        )
        self.render_client = (
            self.vec_client.selfPlayClients[0] if len(self.vec_client.selfPlayClients) > 0 else self.vec_client.clients[0]
        )
        # get the unit type table
        self.utt = json.loads(str(self.render_client.sendUTT()))

    def reset(self):
        self.vec_client.reset([0] * self.num_envs)
        return self.obs

    def step_async(self, actions):
        actions = actions.reshape((self.num_envs, self.width * self.height, self.action_dim))
        np.copyto(self.actions, actions)

    def step_wait(self):
        responses = self.vec_client.gameStep([0] * self.num_envs)
        reward, done = np.array(responses.reward), np.array(responses.done)
        infos = [{"raw_rewards": item} for item in reward]
        # check if it is in evaluation, if not, then change maps
        if len(self.cycle_maps) > 1:
            # check if an environment is done, if done, reset the client, and replace the observation
            for done_idx, d in enumerate(done[:, 0]):
                # bot envs settings
                if done_idx < self.num_bot_envs:
                    if d:
                        self.vec_client.clients[done_idx].mapPath = next(self.next_map)
                        self.vec_client.clients[done_idx].reset(0)
                        # self.obs[done_idx] = self._encode_obs(np.array(response.observation))
                # selfplay envs settings
                else:
                    if d and done_idx % 2 == 0:
                        done_idx -= self.num_bot_envs  # recalibrate the index
                        self.vec_client.selfPlayClients[done_idx // 2].mapPath = next(self.next_map)
                        self.vec_client.selfPlayClients[done_idx // 2].reset()
                        # self.vec_client.selfPlayClients[done_idx // 2].reset()
                        # self.obs[done_idx] = self._encode_obs(np.array(p0_response.observation))
                        # self.obs[done_idx + 1] = self._encode_obs(np.array(p1_response.observation))
        return self.obs, reward @ self.reward_weight, done[:, 0], infos

    def get_action_mask(self):
        self.vec_client.getMasks(0)
        return self.action_mask
