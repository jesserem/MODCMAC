import torch
from torch.distributions import Categorical, Normal
from torchrl.modules import MaskedCategorical
from torch.nn.functional import softmax
from typing import Optional


class BaseDistribution:

    def entropy(self):
        raise NotImplementedError()

    def log_prob(self, actions: torch.Tensor):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()


# class MultiCategorical(BaseDistribution):
#     def __init__(self, logits: Optional[torch.Tensor] = None, probs: Optional[torch.Tensor] = None):
#         # torch.autograd.set_detect_anomaly(True)
#         assert logits is not None or probs is not None, 'Either logits or probs must be provided'
#         if logits is not None and probs is None:
#             probs_old = softmax(logits, dim=-1)
#         else:
#             probs_old = probs
#         mask = torch.zeros_like(probs_old, dtype=torch.bool)
#         mask[-1, :, 2] = True
#         probs_mask = probs_old.masked_fill_(mask, 0)
#
#         self._probs = probs_mask / probs_mask.sum(dim=2, keepdim=True)
#         # print(self._probs)
#         self._distribution = Categorical(probs=self._probs)
#
#     def entropy(self) -> torch.Tensor:
#         return self._distribution.entropy().sum(dim=0)
#
#     def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
#         return self._distribution.log_prob(actions.t()).t().sum(dim=1)
#
#     def sample(self) -> torch.Tensor:
#         return self._distribution.sample().flatten()
#
#     def get_best_action(self) -> torch.Tensor:
#         return self._distribution.probs.argmax(dim=2).flatten()
#
#     @property
#     def probs(self) -> torch.Tensor:
#         return self._distribution.probs


class MultiCategorical(BaseDistribution):
    def __init__(self, logits: Optional[torch.Tensor] = None, probs: Optional[torch.Tensor] = None):
        assert logits is not None or probs is not None, 'Either logits or probs must be provided'
        self._logits = logits
        self._probs = probs
        mask = torch.ones_like(self._logits, dtype=torch.bool)
        mask[-1, :, 2] = False
        self._distribution = MaskedCategorical(logits=self._logits, probs=self._probs, mask=mask)

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
