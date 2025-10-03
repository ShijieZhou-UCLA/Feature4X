# some naive samplers for selecting views
import numpy as np


class BiasViewSampler:
    def __init__(self, min_t, max_t, bias_t, bias_th):
        # the max and min t are included
        self.min_t = min_t
        self.max_t = max_t
        self.bias_t = bias_t
        self.bias_th = bias_th
        return

    def sample_one(self):
        # sample one frame and the adj time
        random_seed = np.random.rand()
        if random_seed < self.bias_th:
            cur = self.bias_t
        else:
            # the uniform mode still can sample the bias time
            cur = np.random.randint(self.min_t, self.max_t + 1)
        if cur == self.max_t:
            adj = cur - 1
        elif cur == self.min_t:
            adj = cur + 1
        else:
            if random_seed < 0.5:
                adj = cur - 1
            else:
                adj = cur + 1
        return cur, adj


# class ContinuesNFrameSampler:
#     def __init__(self):
#         return

#     def sample(self, l, r, n_samples=3):
#         width = r - l + 1
#         if width <= n_samples:
#             return np.arange(l, r + 1).tolist()
#         else:
#             start = l + np.random.randint(0, width + 1 - n_samples)
#             return (np.arange(start, start + n_samples)).tolist()


class RandomSampler:
    def __init__(self):
        return

    def sample(self, l, r, n_samples=1):
        width = r - l + 1
        if width <= n_samples:
            return np.arange(l, r + 1).tolist()
        else:
            choice = np.random.choice(width, n_samples, replace=False) + l
            return choice.tolist()
