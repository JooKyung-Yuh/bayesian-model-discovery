"""STUB ONLY - the development host has no `together` package; autoscilab/llm/client.py imports it at module top.

The Claude path never constructs a Together client. In a real bai run the released venv's own
`together` is imported (the runner adds tests/stub to sys.path only with --stub).
"""
STUB = True


class Together:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("stub Together client: not available in the stub environment")
