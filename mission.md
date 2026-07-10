# tape — reverse-mode autodiff engine with broadcasting
**Tier 3 · Domain DL · Language Python (NumPy) · Est. 6–10 hr · Assigned 2026-07-10**

## The Brief
Build a `Tensor` class that wraps a NumPy array, records the ops applied to it, and can run reverse-mode automatic differentiation over that record to populate `.grad` on every leaf — i.e. you're building the thing `.backward()` actually does inside PyTorch. The catch that makes this a T3 drill and not a toy: your ops must support **broadcasting**, which means the gradient shape doesn't always match the forward shape, and getting that reduction right is where this actually gets interesting.

## Focus from last eval
(first drill — no carried focus)

## Objective
A `Tensor` class, in a file `tape.py` at the repo root, exposing:
- `Tensor(data, requires_grad=False)` — `data` coerced to a NumPy array
- `.data` — the underlying NumPy array
- `.grad` — `None` until a backward pass populates it, then a NumPy array matching `.data`'s shape
- `.backward(grad=None)` — runs reverse-mode autodiff from this tensor back through the graph, accumulating into `.grad` on every leaf with `requires_grad=True`. If `grad` is omitted, assume this tensor is scalar and seed with `1.0`.
- `.zero_grad()` — resets `.grad` to `None`
- Operators: `+`, `-` (both binary and unary neg), `*`, `/`, `@` (matmul), `**` (pow, scalar exponent)
- Methods: `.sum(axis=None, keepdims=False)`, `.reshape(*shape)`, `.transpose(*axes)`, `.exp()`, `.log()`, `.relu()`

Every binary op must support NumPy-style broadcasting between operands of different (but compatible) shapes, and produce correct gradients for each operand's *original* shape.

## Constraints — what you may and may not use
**Allowed:** NumPy for array storage and elementwise/linear-algebra ops, Python stdlib.
**Forbidden:** `torch.autograd` (or any torch use in your implementation), `jax`, the `autograd` package, TensorFlow `GradientTape`, or any other existing autodiff library. Torch appears only inside `tests/`, as the numeric reference oracle — you never import it in `tape.py`.

## Definition of Done — acceptance criteria
Covered by harness (`pytest tests/ -v`):
- [ ] Gradients match `torch.autograd` to 1e-6 tolerance for: chained elementwise ops, broadcast add/mul between mismatched shapes, matmul chains (a 2-layer linear forward), `.sum(axis=...)` reductions, and a diamond-dependency graph (a value used on two branches that later recombine).
- [ ] Calling `.backward()` twice in a row without `.zero_grad()` **accumulates** gradient (matches PyTorch's default) rather than overwriting it.
- [ ] After `.zero_grad()` and a fresh forward pass with mutated leaf data, `.backward()` produces gradients matching a fresh reference computation — i.e. the graph is rebuilt per forward call, not stale from the first pass.
- [ ] A chain of 1000 sequential ops backprops without hitting Python's recursion limit.

Manual check (not automated — self-verify, I'll ask about it in review):
- [ ] README: what this is, the key design decision (how you store the graph, how you handle broadcast-gradient reduction), how to run the example.
- [ ] A runnable example script that fits a tiny 2-layer MLP to XOR using only your `Tensor` and a hand-written SGD loop (no `.step()` abstraction needed — a raw `for p in params: p.data -= lr * p.grad` loop is fine).
- [ ] Repo is clean: no dead code, no commented-out experiments left in.

## Harness
Run: `pytest tests/ -v`
Checks all the numeric/behavioral criteria above against a PyTorch oracle. It imports `from tape import Tensor` — that's the only interface contract it assumes. Manual-check items are listed as a comment at the top of `tests/test_tape.py`. Don't edit the harness; if the interface above genuinely doesn't work for your design, come talk to me and I'll update it.

## Nudges — high-level structure only (no solution)
- Two passes: forward builds the graph (each `Tensor` remembers which op produced it and its parent `Tensor`s), backward walks it. You don't need a full `Function` class hierarchy — a tensor storing its parents plus a "local backward" closure is enough.
- Topological sort before the backward pass. Skipping this is the classic bug: a tensor with two children needs both of its downstream gradients summed *before* it propagates further back, or you'll silently get wrong answers on any diamond-shaped graph.
- Broadcasting breaks the assumption that a gradient has the same shape as its tensor. Treat it as two separate sub-problems: (1) the local derivative formula for the op, computed at the *broadcast* shape, and (2) a shape-reduction step that sums the broadcast gradient back down to each original operand's shape. Solve them independently.
- Build bottom-up: get scalar `add` and `mul` gradient-matching against torch before touching `matmul` or broadcasting at all.

## Stretch goals (optional, extra resume weight)
- A `no_grad()` context manager that suspends graph-building (mirrors `torch.no_grad()`).
- A second-order gradient test: differentiate through your own `backward()` graph to get `d²y/dx²` for `pow`.

## Resources
1. **CS231n — Backprop, Intuitions** (read first) — https://cs231n.github.io/optimization-2/ — builds the mental model of backprop as local gradients chained through a computational graph. This is the "why" before you write any code.
2. **CS231n — Derivatives, Backpropagation, and Vectorization handout** (read when you hit matmul/broadcast) — https://cs231n.stanford.edu/handouts/derivatives.pdf — the Jacobian view of vector/matrix derivatives; directly explains why a broadcast gradient needs axis-sum reduction.
3. **PyTorch — Autograd mechanics notes** — https://docs.pytorch.org/docs/main/notes/autograd.html — the DAG / leaf / grad_fn mental model your engine conceptually mirrors, and documents the default gradient-accumulation-on-repeated-backward behavior this drill's DoD tests for.
4. **karpathy/micrograd** — https://github.com/karpathy/micrograd — **reference implementation, don't open until you've got scalar add/mul working and gradient-matching yourself.** Note it's scalar-only — no broadcasting — so it'll show you the DAG/topo-sort shape but not the actual hard part of this drill.

**Callback:** this is drill #1 — there's no prior project yet, but every future DL drill (manual-backprop MLP, transformer block, attention kernels) builds on top of this `Tensor`. Get the broadcast-gradient reduction genuinely right now and it pays off for months.

## Resume signal
**Bullet:** "Built a reverse-mode automatic differentiation engine from scratch in NumPy supporting broadcasting, matmul, and gradient accumulation, validated to 1e-6 against PyTorch's autograd across chained, branching, and diamond-dependency computation graphs."
**What it signals:** you understand what actually happens inside `.backward()`, not just how to call it — the signal that separates "used PyTorch" from "could help debug PyTorch."

## Submitting
Invoke `/coding-drills review` from the workspace when done (or partially done — honest grades either way). Stopping mid-task? `/coding-drills handoff` snapshots your progress.
