"""The two parts of the agent's run that the report depends on, kept apart from run_task.py so that they can be
tested without the harness: the price of one model call, and the context block placed ahead of the ticket.

Standard library only. run_task.py imports this inside the task container (from /runner); the tests import it
from bench/runner. Added for the context file study (post 5 of "Ask your own repository").
"""

# The context block: the frozen file's text goes in as a template variable, never pasted into the template, so
# that a file containing Jinja syntax such as {{ x }} reaches the agent as written.
CONTEXT_VAR = "context_file_text"
CONTEXT_LABEL = "Contents of AGENTS.md at the repository root:"
CONTEXT_BLOCK = "<context_file>\n" + CONTEXT_LABEL + "\n\n{{ " + CONTEXT_VAR + " }}\n</context_file>\n\n"


def compose_instance_template(template):
    """The harness's instance template with the context block in front of it. The harness's own text, from its
    <pr_description> block on, is kept byte for byte; only the block is added. A run without a context file
    does not call this, so its template is the harness's own."""
    if CONTEXT_VAR in template:
        raise ValueError(f"the harness template already uses {CONTEXT_VAR}; pick another variable name")
    return CONTEXT_BLOCK + template


def prompt_tokens(usage):
    """A request's prompt length as the price list counts it: every input token, cached ones included."""
    return usage.get("input", 0) + usage.get("cache_read", 0) + usage.get("cache_write", 0)


def per_token(price_per_mtok):
    return {k: v / 1e6 for k, v in price_per_mtok.items()}


def call_prices(usage, model_cfg):
    """The per-token prices for one call. A settings file with one flat price list (bench/harness.json) gets it
    for every call, as before; one with a `long_prompt` tier (bench/harness.haiku.json) gets the higher list for
    a call whose prompt is longer than the tier's threshold. Each call is priced on its own."""
    long = model_cfg.get("long_prompt")
    if long and prompt_tokens(usage) > long["above_prompt_tokens"]:
        return per_token(long["price_per_mtok"])
    return per_token(model_cfg["price_per_mtok"])


def call_cost(usage, model_cfg):
    """Dollars for one call, from the usage the gateway reports. For a flat price list the sum is the one
    run_task.py computed before the tiers were added, term by term, so earlier runs cost the same to the cent."""
    price = call_prices(usage, model_cfg)
    return sum(usage[k] * price[k] for k in usage)


def input_price_per_token(usage, model_cfg):
    """The input price of the call's tier, per token: what one more token of prompt costs on that call."""
    return call_prices(usage, model_cfg)["input"]
