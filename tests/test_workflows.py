"""The GitHub Actions workflows.

Workflow YAML is not covered by any other test and fails late and expensively:
a typo in a script path costs a runner minute and a red tick, and a missing
timeout can cost six hours of a monthly budget. These are cheap static checks
for the mistakes that are worth catching before a push rather than after one.
"""

from __future__ import annotations

import re

import pytest
import yaml

from edgeinspect import paths

WORKFLOWS = sorted((paths.ROOT / ".github" / "workflows").glob("*.yml"))

# GitHub's default job timeout is 360 minutes.  Anything near it is a budget
# accident waiting to happen rather than a deliberate choice.
MAX_TIMEOUT_MINUTES = 45


def load(path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def triggers(workflow: dict) -> dict:
    """The ``on:`` block.

    PyYAML resolves a bare ``on`` key to the boolean ``True`` under the YAML 1.1
    rules it implements, which is why this is not simply ``workflow["on"]``.
    """
    if True in workflow:
        return workflow[True]
    return workflow["on"]


def jobs(workflow: dict) -> dict:
    return workflow["jobs"]


def steps(job: dict) -> list:
    return job.get("steps", [])


def run_lines(workflow: dict) -> str:
    """Every ``run:`` body in a workflow, concatenated."""
    out = []
    for job in jobs(workflow).values():
        for step in steps(job):
            if "run" in step:
                out.append(str(step["run"]))
    return "\n".join(out)


def test_there_are_workflows():
    assert WORKFLOWS, "no workflows found under .github/workflows"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_parses_and_has_jobs(path):
    workflow = load(path)
    assert workflow.get("name"), "{0} has no name".format(path.name)
    assert jobs(workflow), "{0} defines no jobs".format(path.name)
    assert triggers(workflow), "{0} has no triggers".format(path.name)


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_job_has_a_sane_timeout(path):
    """The single most important line in a workflow on a metered account.

    Without ``timeout-minutes`` a hung step runs to GitHub's six-hour default,
    which is a fifth of a free month's budget spent on one mistake.
    """
    for name, job in jobs(load(path)).items():
        timeout = job.get("timeout-minutes")
        assert timeout is not None, (
            "{0}: job {1!r} has no timeout-minutes".format(path.name, name))
        assert 0 < int(timeout) <= MAX_TIMEOUT_MINUTES, (
            "{0}: job {1!r} has a timeout of {2} minutes".format(path.name, name, timeout))


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_permissions_are_declared_and_read_only(path):
    """Least privilege, and a guard against a workflow quietly gaining write."""
    workflow = load(path)
    permissions = workflow.get("permissions")
    assert permissions is not None, "{0} does not declare permissions".format(path.name)
    if isinstance(permissions, str):
        assert permissions == "read-all", path.name
    else:
        for scope, level in permissions.items():
            assert level == "read", (
                "{0} grants {1}: {2}; nothing here should need write"
                .format(path.name, scope, level))


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_concurrency_is_configured(path):
    """Every workflow needs a concurrency group; only some should cancel.

    A new push supersedes the previous one, so a push-triggered run should
    cancel its predecessor rather than paying for both.  A scheduled or manual
    run is different: nothing supersedes it, and cancelling one throws away the
    week of evidence or the release bundle somebody asked for.  So the rule is
    about the trigger, not about tidiness.
    """
    workflow = load(path)
    concurrency = workflow.get("concurrency")
    assert concurrency, "{0} has no concurrency group".format(path.name)
    assert concurrency.get("group"), "{0} has no concurrency group name".format(path.name)
    on = triggers(workflow)
    if "push" in on or "pull_request" in on:
        assert concurrency.get("cancel-in-progress") is True, (
            "{0} is push-triggered and does not cancel superseded runs"
            .format(path.name))
    else:
        assert concurrency.get("cancel-in-progress") is False, (
            "{0} would cancel a run nobody superseded".format(path.name))


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_script_a_workflow_runs_exists(path):
    """A typo in a path costs a runner minute and a red tick to discover."""
    body = run_lines(load(path))
    referenced = set(re.findall(r"(?:scripts|pipelines)/[A-Za-z0-9_./-]+\.(?:py|sh)", body))
    assert referenced, "{0} runs no project scripts at all".format(path.name)
    for relative in sorted(referenced):
        assert (paths.ROOT / relative).exists(), (
            "{0} runs {1}, which does not exist".format(path.name, relative))


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_torch_is_installed_from_the_cpu_index(path):
    """The default PyPI wheel pulls 2.5 GB of CUDA that no runner can use."""
    body = run_lines(load(path))
    if "requirements-train.txt" not in body:
        pytest.skip("this workflow does not install torch")
    assert "download.pytorch.org/whl/cpu" in body, (
        "{0} installs torch without pinning the CPU index".format(path.name))


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_pip_caching_is_keyed_on_the_requirement_files(path):
    """An uncached torch install is about a minute of download, every run."""
    for name, job in jobs(load(path)).items():
        for step in steps(job):
            uses = str(step.get("uses", ""))
            if not uses.startswith("actions/setup-python"):
                continue
            with_block = step.get("with", {})
            assert with_block.get("cache") == "pip", (
                "{0}: job {1!r} sets up Python without the pip cache"
                .format(path.name, name))
            assert "requirements" in str(with_block.get("cache-dependency-path", "")), (
                "{0}: job {1!r} caches pip without keying on the requirements"
                .format(path.name, name))


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_uploaded_artifacts_have_a_short_retention(path):
    """Artifact storage is a quota as well as a convenience."""
    for name, job in jobs(load(path)).items():
        for step in steps(job):
            if not str(step.get("uses", "")).startswith("actions/upload-artifact"):
                continue
            with_block = step.get("with", {})
            retention = with_block.get("retention-days")
            assert retention is not None, (
                "{0}: job {1!r} uploads an artifact with the default 90-day retention"
                .format(path.name, name))
            assert int(retention) <= 30, (
                "{0}: job {1!r} keeps an artifact for {2} days"
                .format(path.name, name, retention))


def test_the_push_workflow_runs_the_whole_loop():
    """The assertion the repository is actually about has to be in CI.

    If this starts failing, somebody has removed the one step that checks the
    monitor does not false-alarm, that the gate reaches a decision, and that
    rollback works.
    """
    ci = load(paths.ROOT / ".github" / "workflows" / "ci.yml")
    body = run_lines(ci)
    assert "scripts/run_loop.py" in body, "ci.yml no longer runs the operational loop"
    assert "scripts/check_docs.py" in body, "ci.yml no longer checks the documentation"
    assert "--verify" in body, "ci.yml no longer checks dataset determinism"


def test_the_push_workflow_does_not_claim_a_jetson():
    """Any TensorRT step in a hosted-runner workflow must be a dry run."""
    ci = load(paths.ROOT / ".github" / "workflows" / "ci.yml")
    for line in run_lines(ci).splitlines():
        if "export_tensorrt.py" in line:
            assert "--dry-run" in line, (
                "ci.yml runs the TensorRT builder for real on a runner with no GPU")


def test_no_workflow_hardcodes_a_secret():
    """A cheap guard against a token being pasted into a workflow."""
    suspicious = re.compile(r"(ghp_|github_pat_|gh[ousr]_)[A-Za-z0-9_]{10,}")
    for path in WORKFLOWS:
        text = path.read_text(encoding="utf-8")
        assert not suspicious.search(text), (
            "{0} appears to contain a hardcoded token".format(path.name))


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_python_embedded_in_a_workflow_compiles(path):
    """A syntax error in a heredoc costs a runner minute to discover.

    Two things are checked at once: that the code parses, and that the YAML
    block scalar left the heredoc terminator at column zero.  An indented
    terminator makes the shell swallow the rest of the step.
    """
    import textwrap

    workflow = load(path)
    found = 0
    for job_name, job in jobs(workflow).items():
        for step in steps(job):
            body = step.get("run")
            if not body or "<<'PYTHON'" not in body:
                continue
            blocks = re.findall(r"<<'PYTHON'\n(.*?)\n\s*PYTHON(?:\n|$)", body, re.DOTALL)
            assert blocks, (
                "{0}: step {1!r} opens a PYTHON heredoc that is never closed"
                .format(path.name, step.get("name", job_name)))
            for block in blocks:
                found += 1
                compile(textwrap.dedent(block),
                        "{0}:{1}".format(path.name, step.get("name", job_name)), "exec")
    if not found:
        pytest.skip("this workflow embeds no Python")
