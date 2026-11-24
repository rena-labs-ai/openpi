# rena-task-grammar

The canonical task-string grammar (`compose`, `parse`, the forms, stages and location prepositions), shared by every
repo that writes or reads a task string. Standard library only: a zip Lambda, the pipeline images, the training boxes and
the policy server all import it.

Consumers pin a commit and bump it by hand, in a PR that also carries any change the new grammar needs on their side.

```bash
uv run --with pytest pytest -q tests && uvx ruff@0.17.3 check .
```
