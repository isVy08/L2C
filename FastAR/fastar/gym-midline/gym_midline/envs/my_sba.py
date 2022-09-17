import gym, torch
import numpy as np
import random, copy, os, sys
from sklearn.neighbors import NearestNeighbors

sys.path.append("../../../../")
sys.path.append("../")
sys.path.append("../../")
import classifier_dataset as classifier


class MySBA(gym.Env):
    metadata = {"render.modes": ["human"]}
    """ A custom OpenAI gym for the reduced version of SBA Credit dataset """

    def __init__(self, model_id):
        super(MySBA, self).__init__()
        clf, X_train, X_val, X_test, encoded_columns, immutable_features = classifier.load_model_sba(model_id)
        # Discrete action space
        
        self.action_space = gym.spaces.Discrete(2 * X_train.shape[1])

        low = np.ones(shape=X_train.shape[1]) * -1.0
        high = np.ones(shape=X_train.shape[1])
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float
        )
        self.state = None
        self.name = 'SBA'
        self.dist_lambda = 0.1
        self.immutable_features = [] # immutable_features
        self.dataset = X_test
        self.train_dataset = X_test.to_numpy()
        self.test_dataset = X_test.to_numpy()
        self.state_count = 0
        # self.scaler = scaler
        self.encoded_columns = encoded_columns
        self.classifier = clf
        self.states = {}
        self.states_reverse = {}
        self.no_neighbours = 1
        self.knn_lambda = 0.1
        self.knn = NearestNeighbors(n_neighbors=5, p=1)
        self.knn.fit(self.dataset)
        self.numerical_features = []
        self.seq = -1
        os.environ["SEQ"] = "-1"
        self.undesirable_x = []
        try:
            self.undesirable_x = np.load(
                f"{os.path.dirname(os.path.realpath(__file__))}/../../../datapoints_to_generate_cfes/undesirable_x_mySBA.npy"
            )
            print("Found")
        except:
            undesirable_x = []
            for no, i in enumerate(self.test_dataset):
                if (
                    self.classifier.predict(
                        i.reshape(1, -1)
                    )
                    == 0
                ):
                    undesirable_x.append(tuple(i))
            self.undesirable_x = np.array(undesirable_x)
            # np.save(f"{os.path.dirname(os.path.realpath(__file__))}/../../../datapoints_to_generate_cfes/undesirable_x_SBA.npy", undesirable_x)

        print(
            len(self.undesirable_x), "Total datapoints to run the approach on"
        )
        self.reset()

    def model(self):
        # The probability of belonging to class 1 (the desired class)
        probability_class1 = self.classifier.predict_proba(
            self.state.reshape(1, -1)
        )[0, 1]
        # If the probability of belonging to the desired class is greater than 0.5, then it is a valid CFE.
        if probability_class1 >= 0.5:
            return 100, True
        return probability_class1, False

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.numpy()[0][0]
            assert isinstance(action, (int, np.int64))
            type_ = 1

        elif isinstance(action, np.ndarray):
            if action.shape[0] == 1:
              action = action[0]
              type_ = 1
            else:
              type_ = 2

        elif isinstance(action, (int, np.int64)):
            type_ = 1

        else:
            raise NotImplementedError

        info = {}

        if type_ == 1:
            feature_changing = (
                action // 2
            )  # this is the feature that is changing
            decrease = bool(action % 2)
            if decrease:
                amount = -0.05
            else:
                amount = 0.05

        elif type_ == 2:
            decrease = False
            amount = np.clip(
                action[0], 0, 2 * self.dataset.shape[1]
            )
            if amount < 0:
                decrease = True
            feature = np.clip(
                action[1], 0, 2 * self.dataset.shape[1]
            )
            feature += 1  # casts in 0 to 2 range
            feature_changing = int(
                feature * (self.dataset.shape[1] // 2)
            )  # we need int not round

        else:
            assert False

        reward = -10
        done = False

        for imf in self.immutable_features:
            if imf in self.dataset.iloc[:, feature_changing].name:
                return self.state, reward, done, info


        action_ = amount
        next_state = list(copy.deepcopy(self.state))
        next_state[feature_changing] = self.state[feature_changing] + action_
        knn_dist_loss = self.knn_lambda * self.distance_to_closest_k_points(
            next_state
        )
        assert knn_dist_loss >= 0
        constant = 0  # constant loss for each action

        if decrease:
            if (
                next_state[feature_changing] > -1.0
            ):  # lowest value for a feature is -1.0
                self.state = np.array(next_state)
                reward, done = self.model()
                reward = (
                    reward - constant - knn_dist_loss
                )  # constant cost for each action
            else:
                reward = (
                    -10
                )  # This is the reward in the case of an incorrect action.
                done = False
        else:
            if next_state[feature_changing] < 1.0:  # highest value possible
                self.state = np.array(
                    next_state
                )  # change self.state only if next_state is valid
                reward, done = self.model()
                reward = reward - constant - knn_dist_loss
            else:
                reward = -10
                done = False

        return self.state, reward, done, info

    def distance_to_closest_k_points(self, state):
        state = np.array([state]).reshape(1, -1)
        nearest_dist, nearest_points = self.knn.kneighbors(
            state, self.no_neighbours, return_distance=True
        )
        return np.mean(nearest_dist)

    def reset(self):
        seq = int(os.environ["SEQ"])
        if len(self.undesirable_x) == 0:
            return
        # This is used during training of the agent.
        if seq == -1:
            idx = random.randrange(self.train_dataset.shape[0])
            self.state = self.train_dataset[idx]
        # This is used during evaluation of a trained agent
        else:
            self.state = np.array(self.undesirable_x[seq]).reshape(1, -1)[0]
        return self.state

    def render(self, mode="human", close=False):
        print(f"State: {self.state}")


class MySBA1(MySBA):
    def __init__(self, enable_render=True):
        super(MySBA1, self).__init__(1)

class MySBA2(MySBA):
    def __init__(self, enable_render=True):
        super(MySBA2, self).__init__(2)

class MySBA3(MySBA):
    def __init__(self, enable_render=True):
        super(MySBA3, self).__init__(3)


class MySBA4(MySBA):
    def __init__(self, enable_render=True):
        super(MySBA4, self).__init__(4)


class MySBA5(MySBA):
    def __init__(self, enable_render=True):
        super(MySBA5, self).__init__(5)


if __name__ == "__main__":
    x = MySBA1()
    print(x.step(1))
    print(x.step(5))


