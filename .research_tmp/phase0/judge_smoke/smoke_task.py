# -*- coding: utf-8 -*-
"""Phase 0.4 judge smoke: official sqa scorer × GLM-5.3 (Paratera).

Memorized solver emits a hand-crafted CS2-shaped report for dev[0]
(dropout side effects) — content is real-ish but the point is JUDGE
plumbing: response_schema json_schema compat, retry path, four-facet
score production, per-call latency.

Run:
  inspect eval smoke_task.py::sqa_memorized_glm
Env: GLM_API_KEY / GLM_BASE_URL (Paratera), HF_TOKEN.
"""
from inspect_ai import Task, task
from inspect_ai.solver import solver

from astabench.evals.sqa.task import sqa

REPORT = """{
  "sections": [
    {
      "title": "Positive side effects",
      "text": "Beyond regularization, dropout produces several desirable side effects. Training with dropout is equivalent to training an ensemble of subnetworks with shared weights, which improves model robustness and reduces overfitting to specific patterns [1]. At test time, Monte Carlo dropout — leaving dropout active during inference and averaging multiple stochastic forward passes — provides epistemic uncertainty estimates without architectural changes [1]. Models trained with dropout also show improved resilience to input noise, feature perturbations, and adversarial examples, because individual neurons become less reliant on co-adapted feature combinations [2].",
      "citations": [
        {
          "id": "[1]",
          "snippets": ["Deeply-supervised nets. Dropout can be interpreted as a form of model averaging. Monte Carlo sampling of dropout masks at test time approximates Bayesian inference."],
          "title": "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning",
          "metadata": {"year": 2016, "authors": "Gal, Y. & Ghahramani, Z."}
        },
        {
          "id": "[2]",
          "snippets": ["A simple way to prevent neural networks from overfitting. Each hidden unit must learn features that work with many random subsets of other units."],
          "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
          "metadata": {"year": 2014, "authors": "Srivastava, N. et al."}
        }
      ]
    },
    {
      "title": "Negative side effects",
      "text": "Dropout also carries costs. The noise introduced during training generally necessitates longer training to reach convergence, increasing training time and compute [2]. There is a non-negligible inconsistency between training and inference: subnetworks are optimized during training, while the full scaled network is used at inference [2]. Monte Carlo dropout requires multiple forward passes (often around one hundred) to estimate uncertainty, which can multiply inference cost substantially [1].",
      "citations": [
        {
          "id": "[1]",
          "snippets": ["Monte Carlo dropout requires multiple forward passes, typically on the order of one hundred, to estimate predictive uncertainty."],
          "title": "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning",
          "metadata": {"year": 2016, "authors": "Gal, Y. & Ghahramani, Z."}
        },
        {
          "id": "[2]",
          "snippets": ["Dropout roughly doubles the number of iterations required to converge compared to standard SGD."],
          "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
          "metadata": {"year": 2014, "authors": "Srivastava, N. et al."}
        }
      ]
    }
  ]
}"""


@solver
def memorized_solver():
    async def solve(state, generate):
        state.output.completion = REPORT
        return state
    return solve


@task
def sqa_memorized_glm() -> Task:
    t = sqa(scorer_model="openai-api/glm/GLM-5.3", limit=1,
            with_search_tools=False)
    t.solver = memorized_solver()
    return t
