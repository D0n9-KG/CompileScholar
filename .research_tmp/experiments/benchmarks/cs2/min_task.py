from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.scorer import match

@task
def min_glm() -> Task:
    return Task(dataset=[Sample(input="Reply with exactly: hello world", id="1")],
                solver=generate(),
                scorer=match())
