# The fairness screen

Give a labeller (a person, or a fresh model session that has not seen any run result) the packets that `bench.py packets` writes, and this rule. Labels go in a JSON-lines file, one object per task with `task_id`, `label` and `reason`, for `bench.py report --labels`.

Reading only the issue as filed and the withheld test changes, could a competent developer who had not seen the tests write a fix that passes the fail-to-pass tests?

- **fair**: the tests check behaviour the issue states or clearly implies.
- **unfair**: at least one fail-to-pass test depends on something the issue neither states nor implies and a reasonable fix could do differently: a new public or private name, module or signature that the test imports or calls; the exact wording of a new or changed message; one choice among several reasonable behaviours.
- **unclear**: it cannot be told from the packet.

Judge the fail-to-pass tests only, not style or unrelated test edits. A test asserting on a message that the issue quotes or clearly asks for is fair; one asserting on wording the issue does not give is unfair. A test importing a helper the issue never mentions is unfair. A behaviour the issue describes as expected is fair even if the test's way of checking it is elaborate. Give one sentence of reason per task, naming the specific thing in the tests that decides it.

Fix the rule before anyone labels, keep the labeller away from run results, and check a few labels by hand, drawn at random.
