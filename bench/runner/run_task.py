#!/opt/harness/bin/python
"""Runs mini-swe-agent on one task, inside the task container, as root. bench.py starts it and reads /out.

Reads the gateway key as one JSON line on standard input, so the key is never in a file or in the container's
environment. Prepares /testbed with inside.prepare, then runs the harness with its own swebench.yaml config and
only the overrides in /bench/harness.json. Every command the agent chooses runs as the unprivileged user
`agent` with a clean environment: no key, no proxy settings, no access to /out. The harness alone talks to the
model, through the bench-egress proxy, the only way out of the container's network.

Writes /out/trajectory.json (the harness's own record), /out/submission.diff (every change in the working tree
against the base commit, without the harness's patch.txt), /out/result.json and /out/harness.log.

Two additions for the context file study, both off unless meta.json asks for them, so a run without them gets
the same prompt as before: `context` (a frozen context file in /task, placed in its own block ahead of the
ticket by agent_parts.compose_instance_template and handed to the template as a variable) and `task_text` with
`collect` (context.py generate: the initialisation prompt as the task, and the files the agent wrote copied to
/out/collected). Cost is counted per call at the price tier of that call's prompt length (agent_parts.call_cost).
"""
import json
import os
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, "/runner")
import inside  # noqa: E402
import agent_parts  # noqa: E402

CFG = json.loads(Path("/bench/harness.json").read_text(encoding="utf-8"))
META = json.loads(Path("/task/meta.json").read_text(encoding="utf-8"))
OUT = Path("/out")

# The proxy is set for this process only, before LiteLLM is imported; the agent's commands never see it.
os.environ["HTTPS_PROXY"] = os.environ["https_proxy"] = CFG["environment"]["proxy"]
os.environ["NO_PROXY"] = os.environ["no_proxy"] = ""
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import yaml  # noqa: E402
from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.config import get_config_path  # noqa: E402
from minisweagent.environments.local import LocalEnvironment  # noqa: E402
from minisweagent.models.litellm_model import LitellmModel  # noqa: E402


class AgentUserEnvironment(LocalEnvironment):
    """mini-swe-agent's local environment, with each command run as the agent user in a clean environment."""

    def __init__(self, *, agent_env, user, **kw):
        super().__init__(**kw)
        self.agent_env, self.user = agent_env, user

    def execute(self, action, cwd="", *, timeout=None):
        clean = " ".join(shlex.quote(f"{k}={v}") for k, v in self.agent_env.items())
        command = (f"{inside.RUNUSER} -u {self.user} -- env -i {clean} bash -c "
                   f"{shlex.quote(action.get('command', ''))}")
        return super().execute({**action, "command": command}, cwd, timeout=timeout)


def usage_of(response):
    u = getattr(response, "usage", None)
    if u is None:
        return {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    details = getattr(u, "prompt_tokens_details", None)
    cache_read = (getattr(details, "cached_tokens", 0) if details else 0) or 0
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    prompt = getattr(u, "prompt_tokens", 0) or 0
    return {"input": max(prompt - cache_read - cache_write, 0), "output": getattr(u, "completion_tokens", 0) or 0,
            "cache_read": cache_read, "cache_write": cache_write}


class GatewayModel(LitellmModel):
    """LiteLLM has no price for a model reached through the gateway's OpenAI-compatible endpoint, so cost is
    counted here from the usage the gateway reports, at the list prices in harness.json (the higher tier for a
    call whose prompt passes the tier's threshold, when the settings file has one)."""

    def _calculate_cost(self, response):
        u = usage_of(response)
        return {"cost": agent_parts.call_cost(u, CFG["model"]), "usage": u}

    def serialize(self):
        """The harness saves its model settings in the trajectory; the key is passed in them, so it is removed."""
        data = super().serialize()
        kwargs = data.get("info", {}).get("config", {}).get("model", {}).get("model_kwargs", {})
        if "api_key" in kwargs:
            kwargs["api_key"] = "[redacted]"
        return data


def main():
    key = json.loads(sys.stdin.readline())["api_key"]
    log = (OUT / "harness.log").open("w", encoding="utf-8")
    t0 = time.time()
    result = {"task_id": META["task_id"], "exit_status": "", "cost": 0.0, "api_calls": 0}
    try:
        inside.prepare(META)
        base = yaml.safe_load(get_config_path(CFG["base_config"]).read_text(encoding="utf-8"))
        agent_cfg = {**base["agent"], **CFG["agent"], "output_path": OUT / "trajectory.json"}
        env_base = {k: v for k, v in (base.get("environment", {}).get("env") or {}).items() if k != "BASH_ENV"}
        agent_env = {"PATH": "/opt/testbed/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/home/agent",
                     "LANG": "C.UTF-8", **env_base, **META.get("env", {})}
        env = AgentUserEnvironment(cwd=CFG["environment"]["cwd"], timeout=CFG["environment"]["timeout"],
                                   agent_env=agent_env, user=CFG["environment"]["agent_user"])
        mcfg = {k: v for k, v in (base.get("model") or {}).items() if k not in ("model_class", "model_name")}
        mcfg["model_kwargs"] = {**mcfg.get("model_kwargs", {}), "api_base": CFG["model"]["api_base"],
                                "api_key": key}
        model = GatewayModel(model_name=CFG["model"]["model_name"], cost_tracking="ignore_errors", **mcfg)
        extra_vars = {}
        if META.get("context"):  # the frozen file, as a variable: never pasted into the template
            agent_cfg["instance_template"] = agent_parts.compose_instance_template(agent_cfg["instance_template"])
            extra_vars[agent_parts.CONTEXT_VAR] = Path(META["context"]["path"]).read_text(encoding="utf-8")
        agent = DefaultAgent(model, env, **agent_cfg)
        if META.get("task_text") is not None:
            task = META["task_text"]
        else:
            task = CFG["task_preamble"].format(repo=META["repo"]) + "\n\n" + META["issue_title"] + "\n\n" + META["issue_body"]
        out = agent.run(task=task, **extra_vars)
        result.update(exit_status=out.get("exit_status", ""), cost=agent.cost, api_calls=agent.n_calls)
        tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        for m in agent.messages:
            for k, v in ((m.get("extra") or {}).get("usage") or {}).items():
                tokens[k] = tokens.get(k, 0) + v
        result["tokens"] = tokens
    except Exception as e:
        result.update(exit_status=result["exit_status"] or "error", error=repr(e)[:1500])
        try:
            result.update(cost=agent.cost, api_calls=agent.n_calls)  # noqa: F821
        except NameError:
            pass
    finally:
        env_git = {"HOME": "/home/agent", "PATH": "/usr/bin:/bin"}
        try:
            inside.sh(["git", "add", "-A"], user="agent", env=env_git, check=False)
            diff = inside.sh(["git", "diff", "--cached", "--binary", "HEAD", "--", ".", ":(exclude)patch.txt"],
                             user="agent", env=env_git, check=False).stdout
        except Exception as e:  # recorded, the score step treats an empty diff as not resolved
            diff = ""
            result["diff_error"] = repr(e)[:500]
        (OUT / "submission.diff").write_text(diff, encoding="utf-8")
        for name in META.get("collect") or []:  # read as root: the agent user cannot write to /out
            src = inside.TESTBED / name
            if src.is_file():
                (OUT / "collected").mkdir(exist_ok=True)
                (OUT / "collected" / name).write_bytes(src.read_bytes())
        result.update(submission_bytes=len(diff.encode()), wall_s=round(time.time() - t0, 1),
                      capped=result.get("exit_status") == "LimitsExceeded")
        (OUT / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        log.write(json.dumps(result, default=str) + "\n")
        log.close()
        print(json.dumps({k: result.get(k) for k in ("task_id", "exit_status", "cost", "api_calls", "wall_s")},
                         default=str))


if __name__ == "__main__":
    main()
