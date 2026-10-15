# Figures

The README opens with an ASCII diagram rather than an image, deliberately: it is
diffable, it survives a terminal, and it does not go stale silently the way a
rendered picture does when the code under it changes.

`scripts/run_loop.py` writes the run it describes to `results/loop.json`, and
`make demo` prints the same timeline to stdout, so the diagram and the thing it
depicts are checkable against each other.
