import torch
from torch.distributions import Categorical, Normal
from typing import Optional


class BaseDistribution:

    def entropy(self):
        raise NotImplementedError()

    def log_prob(self, actions: torch.Tensor):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()


class MultiCategorical(BaseDistribution):
    def __init__(self, logits: Optional[torch.Tensor] = None, probs: Optional[torch.Tensor] = None):
        assert logits is not None or probs is not None, 'Either logits or probs must be provided'
        self._logits = logits
        self._probs = probs
        self._distribution = Categorical(logits=self._logits, probs=self._probs)

    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(dim=0)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(actions.t()).t().sum(dim=1)

    def sample(self) -> torch.Tensor:
        return self._distribution.sample().flatten()

    def get_best_action(self) -> torch.Tensor:
        return self._distribution.probs.argmax(dim=2).flatten()

    @property
    def probs(self) -> torch.Tensor:
        return self._distribution.probs


class MultiNormal(BaseDistribution):
    def __init__(self, logits: torch.Tensor):
        self._mu = logits[:, :, 0]
        self._var = logits[:, :, 1]
        self._distribution = Normal(self._mu, self._var)

    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(dim=0)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(actions.t()).t().sum(dim=1)

    def sample(self) -> torch.Tensor:
        return self._distribution.sample().flatten()

    def get_best_action(self) -> torch.Tensor:
        return self._mu.flatten()
