import torch
import numpy as np
from abc import ABC, abstractmethod 


class RandomParams(ABC):

    def __init__(self, seed): 
        self.seed = seed

    @abstractmethod
    def sample(self, **kwargs): 
        pass

    def _generators(self, n : int = 3):
        children = np.random.SeedSequence(self.seed).spawn(n)
        gens = []
        for c in children:
            s = int(c.generate_state(1, dtype=np.uint64)[0])
            gens.append(torch.Generator(device="cpu").manual_seed(s))
        return gens


