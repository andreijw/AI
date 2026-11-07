"""Tests for the machine-wide run queue (.claude/skills/experiment-runner/scripts/run_queue.py).

The script is not part of the package, so it is loaded from its path. Jobs are tiny
``python -c`` programs and the queue lives in ``tmp_path``.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the run queue needs fcntl")

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / ".claude"
    / "skills"
    / "experiment-runner"
    / "scripts"
    / "run_queue.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_queue", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_queue"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rq(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The module, with the queue directory in tmp_path and 2 slots."""
    monkeypatch.setenv("RL_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("RL_QUEUE_SLOTS", "2")
    monkeypatch.setenv("RL_QUEUE_POLL", "0.05")
    module = _load()
    # Commands that notice queued jobs start a worker; here that is recorded, not done.
    module.real_spawn_worker = module.spawn_worker
    module.spawned = []
    monkeypatch.setattr(module, "spawn_worker", lambda queue: module.spawned.append(1))
    return module


def _bounded(call: Callable[[], int], timeout: float = 30.0) -> int:
    """Run ``call`` in a thread and fail, rather than hang, if it does not return in time."""
    result: list[int] = []
    thread = threading.Thread(target=lambda: result.append(call()), daemon=True)
    thread.start()
    thread.join(timeout)
    assert result, f"did not return within {timeout} s"
    return result[0]


def _submit(rq: ModuleType, name: str, code: str, item: str = "t", *extra: str) -> None:
    argv = ["submit", "--name", name, "--item", item, "--no-worker", *extra]
    assert rq.main([*argv, "--", sys.executable, "-c", code]) == 0


def _jobs(rq: ModuleType) -> list[dict]:
    return rq.Queue(rq.queue_dir()).jobs()


def _max_overlap(jobs: list[dict]) -> int:
    events = [(j["start"], 1) for j in jobs] + [(j["end"], -1) for j in jobs]
    running = peak = 0
    for _, delta in sorted(events, key=lambda e: (e[0], e[1])):  # an end sorts before a start
        running += delta
        peak = max(peak, running)
    return peak


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def test_slot_cap_is_never_exceeded(rq: ModuleType) -> None:
    for i in range(5):
        _submit(rq, f"j{i}", "import time; time.sleep(0.25)")
    assert rq.main(["run"]) == 0
    jobs = _jobs(rq)
    assert [j["state"] for j in jobs] == ["done"] * 5
    assert _max_overlap(jobs) == 2


def test_a_second_worker_exits_at_once_while_one_holds_the_lock(rq: ModuleType) -> None:
    queue = rq.Queue(rq.queue_dir())
    _submit(rq, "waiting", "pass")
    held = rq.try_worker_lock(queue)
    assert held is not None
    assert rq.main(["run"]) == 0  # returns without running anything
    assert _jobs(rq)[0]["state"] == "queued"
    rq.release_worker_lock(held)


def test_queued_jobs_start_in_submission_order(rq: ModuleType) -> None:
    for i in range(6):
        _submit(rq, f"j{i}", "pass")
    rq.main(["run"])
    started = sorted(_jobs(rq), key=lambda j: j["start"])
    assert [j["name"] for j in started[:2]] == ["j0", "j1"]
    assert [j["name"] for j in _jobs(rq)] == [f"j{i}" for i in range(6)]


def test_failed_job_records_exit_code(rq: ModuleType) -> None:
    _submit(rq, "bad", "import sys; sys.exit(3)")
    assert rq.main(["run"]) == 0
    (job,) = _jobs(rq)
    assert job["state"] == "failed"
    assert job["exit_code"] == 3


def test_job_gets_single_thread_env_and_its_own_process_group(
    rq: ModuleType, tmp_path: Path
) -> None:
    out = tmp_path / "out.txt"
    code = (
        f"import os; open({str(out)!r}, 'w').write(' '.join([os.environ['OPENBLAS_NUM_THREADS'],"
        " os.environ['OMP_NUM_THREADS'], os.environ['MKL_NUM_THREADS'], os.environ['X'],"
        " str(os.getpgrp() == os.getppid())]))"
    )
    _submit(rq, "env", code, "t", "--env", "X=7")
    rq.main(["run"])
    assert out.read_text() == "1 1 1 7 True"


def test_cancel_kills_the_job_and_its_child(rq: ModuleType, tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    code = (
        "import subprocess, time; c = subprocess.Popen(['sleep', '60']);"
        f" open({str(pid_file)!r}, 'w').write(str(c.pid)); time.sleep(60)"
    )
    _submit(rq, "spawner", code)
    worker = subprocess.Popen([sys.executable, str(SCRIPT), "run"], env=os.environ.copy())
    try:
        deadline = time.time() + 10
        while not (pid_file.exists() and pid_file.read_text()) and time.time() < deadline:
            time.sleep(0.05)
        child = int(pid_file.read_text())
        assert _alive(child)
        job_id = _jobs(rq)[0]["id"]
        assert rq.main(["cancel", job_id]) == 0
        deadline = time.time() + 5
        while _alive(child) and time.time() < deadline:
            time.sleep(0.05)
        assert not _alive(child)
        assert worker.wait(timeout=10) == 0
    finally:
        worker.kill()
    assert _jobs(rq)[0]["state"] == "cancelled"


def test_cancel_by_item_cancels_queued_jobs(rq: ModuleType) -> None:
    _submit(rq, "a", "pass", "keep")
    _submit(rq, "b", "pass", "drop")
    assert rq.main(["cancel", "--item", "drop"]) == 0
    assert {j["name"]: j["state"] for j in _jobs(rq)} == {"a": "queued", "b": "cancelled"}


def test_wait_returns_nonzero_after_a_failure(rq: ModuleType) -> None:
    _submit(rq, "ok", "pass", "good")
    _submit(rq, "bad", "raise SystemExit(1)", "mixed")
    _submit(rq, "ok2", "pass", "mixed")
    rq.main(["run"])
    assert _bounded(lambda: rq.main(["wait", "--item", "good"])) == 0
    assert _bounded(lambda: rq.main(["wait", "--item", "mixed"])) == 1


def test_stale_running_record_is_reclaimed(rq: ModuleType, capsys: pytest.CaptureFixture) -> None:
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    queue = rq.Queue(rq.queue_dir())
    _submit(rq, "ghost", "pass")
    job = queue.jobs()[0]  # the only job so far
    job.update(state="running", pid=dead.pid, pgid=dead.pid, worker=dead.pid, start=time.time())
    queue.write(job)
    _submit(rq, "next", "pass")
    _submit(rq, "after", "pass")
    # Two slots, one taken by the ghost: nothing may overrun, and the ghost must be freed.
    rq.main(["run"])
    states = {j["name"]: j["state"] for j in _jobs(rq)}
    assert states == {"ghost": "failed", "next": "done", "after": "done"}
    assert "lost" in next(j for j in _jobs(rq) if j["name"] == "ghost")["error"]
    capsys.readouterr()
    assert rq.main(["status"]) == 0
    assert "ghost" in capsys.readouterr().out


def test_concurrent_submitters_lose_no_job(rq: ModuleType, tmp_path: Path) -> None:
    env = {**os.environ, "RL_QUEUE_DIR": str(tmp_path / "queue")}

    def submit(tag: str) -> None:
        for i in range(3):
            cmd = [sys.executable, str(SCRIPT), "submit", "--name", f"{tag}{i}", "--item", tag]
            subprocess.run([*cmd, "--no-worker", "--", "true"], env=env, check=True)

    threads = [threading.Thread(target=submit, args=(tag,)) for tag in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    jobs = _jobs(rq)
    assert len(jobs) == 6
    assert len({j["id"] for j in jobs}) == 6
    assert {j["name"] for j in jobs} == {f"{t}{i}" for t in "ab" for i in range(3)}


def test_status_lists_one_line_per_job_and_filters_by_item(
    rq: ModuleType, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _submit(rq, "one", "pass", "x")
    _submit(rq, "two", "pass", "y")
    capsys.readouterr()
    assert rq.main(["status", "--item", "x"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and "one" in lines[0] and "queued" in lines[0]


# --- worker lock, auto-start and the lost wakeup ---------------------------------------------


def test_worker_rechecks_the_queue_after_releasing_its_lock(rq: ModuleType) -> None:
    """A job submitted in the window after the worker decided it was idle is still run."""
    _submit(rq, "first", "pass")
    calls: list[int] = []

    def late_submit() -> None:
        probe = rq.try_worker_lock(rq.Queue(rq.queue_dir()))
        assert probe is not None, "the worker must have released worker.lock by now"
        rq.release_worker_lock(probe)
        if not calls:
            calls.append(1)
            _submit(rq, "late", "pass")  # the submitter saw the lock held, so spawned nobody

    rq.before_idle_recheck = late_submit
    assert rq.main(["run"]) == 0
    assert {j["name"]: j["state"] for j in _jobs(rq)} == {"first": "done", "late": "done"}


def test_submit_spawns_a_worker_only_when_nobody_holds_the_lock(
    rq: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned = rq.spawned
    queue = rq.Queue(rq.queue_dir())
    held = rq.try_worker_lock(queue)
    assert held is not None
    assert rq.main(["submit", "--name", "a", "--item", "t", "--", "true"]) == 0
    assert spawned == []
    rq.release_worker_lock(held)
    assert rq.main(["submit", "--name", "b", "--item", "t", "--", "true"]) == 0
    assert spawned == [1]


def test_auto_started_worker_is_unique_and_exits_when_idle(rq: ModuleType, tmp_path: Path) -> None:
    env = {**os.environ, "RL_QUEUE_DIR": str(tmp_path / "queue")}
    code = "import time; time.sleep(0.4)"
    for i in range(3):
        cmd = [sys.executable, str(SCRIPT), "submit", "--name", f"a{i}", "--item", "auto"]
        subprocess.run(
            [*cmd, "--", sys.executable, "-c", code], env=env, check=True, capture_output=True
        )
    assert _bounded(lambda: rq.main(["wait", "--item", "auto"])) == 0
    jobs = _jobs(rq)
    assert [j["state"] for j in jobs] == ["done"] * 3
    assert len({j["worker"] for j in jobs}) == 1
    queue = rq.Queue(rq.queue_dir())
    deadline = time.time() + 5
    lock = None
    while lock is None and time.time() < deadline:
        lock = rq.try_worker_lock(queue)  # free once the worker has exited
        time.sleep(0.05)
    assert lock is not None
    rq.release_worker_lock(lock)


def test_wait_starts_a_worker_for_queued_jobs(
    rq: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rq, "spawn_worker", rq.real_spawn_worker)
    _submit(rq, "orphan", "pass", "w")
    assert _bounded(lambda: rq.main(["wait", "--item", "w"])) == 0
    assert _jobs(rq)[0]["state"] == "done"


# --- the stale rule, pid reuse and the real exit code -------------------------------------------


def _make_running(rq: ModuleType, **fields: object) -> tuple[object, dict]:
    queue = rq.Queue(rq.queue_dir())
    _submit(rq, "r", "pass")
    job = queue.jobs()[0]
    job.update(state="running", start=time.time(), **fields)
    queue.write(job)
    return queue, job


def _dead_pid() -> int:
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    return dead.pid


def test_job_dead_but_worker_alive_stays_running(rq: ModuleType) -> None:
    queue, job = _make_running(rq, pid=_dead_pid(), pgid=_dead_pid(), worker=os.getpid())
    with queue.locked():
        queue.reclaim_stale()
    assert queue.jobs()[0]["state"] == "running"


def test_worker_dead_but_job_alive_stays_running(rq: ModuleType) -> None:
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        queue, job = _make_running(
            rq,
            pid=sleeper.pid,
            pgid=sleeper.pid,
            worker=_dead_pid(),
            pid_start=rq.proc_start(sleeper.pid),
        )
        with queue.locked():
            queue.reclaim_stale()
        assert queue.jobs()[0]["state"] == "running"
    finally:
        sleeper.kill()
        sleeper.wait()


def test_reused_pid_is_not_taken_for_the_job(rq: ModuleType) -> None:
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        queue, job = _make_running(
            rq, pid=sleeper.pid, pgid=sleeper.pid, worker=_dead_pid(), pid_start="1"
        )
        with queue.locked():
            queue.reclaim_stale()
        assert queue.jobs()[0]["state"] == "failed"
    finally:
        sleeper.kill()
        sleeper.wait()


@pytest.mark.parametrize(("written", "state", "code"), [("0", "done", 0), ("5", "failed", 5)])
def test_stale_job_that_finished_keeps_its_real_exit_code(
    rq: ModuleType, written: str, state: str, code: int
) -> None:
    queue, job = _make_running(rq, pid=_dead_pid(), pgid=_dead_pid(), worker=_dead_pid())
    (queue.root / "exit" / job["id"]).write_text(written)
    with queue.locked():
        queue.reclaim_stale()
    got = queue.jobs()[0]
    assert (got["state"], got["exit_code"]) == (state, code)


def test_job_finishing_after_its_worker_was_killed_keeps_its_exit_code(
    rq: ModuleType, tmp_path: Path
) -> None:
    _submit(rq, "orphaned", "import sys, time; time.sleep(1.0); sys.exit(4)")
    worker = subprocess.Popen([sys.executable, str(SCRIPT), "run"])
    try:
        deadline = time.time() + 10
        while _jobs(rq)[0]["state"] != "running" and time.time() < deadline:
            time.sleep(0.05)
        worker.kill()
        worker.wait()
        pid = _jobs(rq)[0]["pid"]
        while rq.pid_alive(pid) and time.time() < deadline:  # the job has really finished
            time.sleep(0.05)
        assert rq.main(["status"]) == 0
    finally:
        worker.kill()
    job = _jobs(rq)[0]
    assert (job["state"], job["exit_code"]) == ("failed", 4)


# --- cancel ----------------------------------------------------------------------------------


def _start_worker(rq: ModuleType) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(SCRIPT), "run"], env=os.environ.copy())


def _wait_for(path: Path) -> int:
    deadline = time.time() + 10
    while not (path.exists() and path.read_text()) and time.time() < deadline:
        time.sleep(0.05)
    return int(path.read_text())


_IGNORE = "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN)"


def test_cancel_escalates_to_sigkill_for_a_job_that_ignores_sigterm(
    rq: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rq, "KILL_GRACE_SECONDS", 0.3)
    pid_file = tmp_path / "job.pid"
    code = f"{_IGNORE}; import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    _submit(rq, "stubborn", code)
    worker = _start_worker(rq)
    try:
        pid = _wait_for(pid_file)
        assert rq.main(["cancel", _jobs(rq)[0]["id"]]) == 0
        assert not _alive(pid)
        assert worker.wait(timeout=10) == 0
    finally:
        worker.kill()
    assert _jobs(rq)[0]["state"] == "cancelled"


def test_cancel_kills_a_descendant_that_ignores_sigterm(
    rq: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rq, "KILL_GRACE_SECONDS", 0.3)
    pid_file = tmp_path / "child.pid"
    child = f"{_IGNORE}; import time; time.sleep(60)"
    code = (
        f"import subprocess, sys, time; c = subprocess.Popen([sys.executable, '-c', {child!r}]);"
        f" open({str(pid_file)!r}, 'w').write(str(c.pid)); time.sleep(60)"
    )
    _submit(rq, "parent", code)
    worker = _start_worker(rq)
    try:
        pid = _wait_for(pid_file)
        assert rq.main(["cancel", _jobs(rq)[0]["id"]]) == 0
        assert not _alive(pid)
        worker.wait(timeout=10)
    finally:
        worker.kill()


def test_cancel_keeps_the_slot_until_the_processes_are_gone(
    rq: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rq, "KILL_GRACE_SECONDS", 1.5)
    pid_file = tmp_path / "job.pid"
    code = f"{_IGNORE}; import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    _submit(rq, "stubborn", code)
    worker = _start_worker(rq)
    try:
        _wait_for(pid_file)
        job_id = _jobs(rq)[0]["id"]
        canceller = threading.Thread(target=rq.main, args=(["cancel", job_id],))
        canceller.start()
        deadline = time.time() + 10
        while not _jobs(rq)[0].get("cancelling") and time.time() < deadline:
            time.sleep(0.02)  # the flag is set before the signal; the grace period is 1.5 s
        job = _jobs(rq)[0]
        assert job["state"] == "running" and job["cancelling"] is True
        canceller.join(timeout=10)
        worker.wait(timeout=10)
    finally:
        worker.kill()
    assert _jobs(rq)[0]["state"] == "cancelled"


def test_cancel_leaves_a_finished_job_alone(rq: ModuleType) -> None:
    _submit(rq, "fin", "pass")
    rq.main(["run"])
    rq.main(["cancel", _jobs(rq)[0]["id"]])
    assert _jobs(rq)[0]["state"] == "done"


def test_cancel_does_not_signal_a_reused_pid(rq: ModuleType) -> None:
    """The pid is alive but is a different process (start time differs): not our job's group."""
    bystander = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        _make_running(rq, pid=bystander.pid, pgid=bystander.pid, pid_start="1", worker=_dead_pid())
        rq.main(["cancel", _jobs(rq)[0]["id"]])
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait()


def test_cancel_of_a_dead_group_just_closes_the_record(rq: ModuleType) -> None:
    _make_running(rq, pid=_dead_pid(), pgid=_dead_pid(), worker=_dead_pid())
    assert rq.main(["cancel", _jobs(rq)[0]["id"]]) in (0, 1)
    assert _jobs(rq)[0]["state"] in ("failed", "cancelled")


# --- usage errors ----------------------------------------------------------------------------


def test_submit_without_a_command_is_a_usage_error(rq: ModuleType) -> None:
    assert rq.main(["submit", "--name", "n", "--item", "i", "--no-worker", "--"]) == 2


def test_env_without_equals_is_a_usage_error(rq: ModuleType) -> None:
    argv = ["submit", "--name", "n", "--item", "i", "--no-worker", "--env", "BAD", "--", "true"]
    assert rq.main(argv) == 2


def test_invalid_slot_count_is_a_usage_error(
    rq: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RL_QUEUE_SLOTS", "lots")
    assert rq.main(["run"]) == 2


# --- failed writes ---------------------------------------------------------------------------


def _fail_first_write(rq: ModuleType, monkeypatch: pytest.MonkeyPatch, states: tuple[str, ...]):
    real = rq.Queue.write
    failures: list[int] = []

    def write(self: object, job: dict) -> None:
        if job["state"] in states and not failures:
            failures.append(1)
            raise OSError(28, "No space left on device")
        real(self, job)

    monkeypatch.setattr(rq.Queue, "write", write)
    return failures


def test_job_is_not_run_twice_when_the_running_record_cannot_be_written(
    rq: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "ran.txt"
    code = f"import time; time.sleep(0.3); open({str(marker)!r}, 'a').write('x')"
    _submit(rq, "once", code)
    failures = _fail_first_write(rq, monkeypatch, ("running",))
    assert rq.main(["run"]) == 0
    assert failures == [1]
    assert _jobs(rq)[0]["state"] == "done"
    assert marker.read_text() == "x"


def test_worker_survives_a_failed_write_when_reaping(
    rq: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _submit(rq, "reap", "pass")
    failures = _fail_first_write(rq, monkeypatch, ("done",))
    assert rq.main(["run"]) == 0
    assert failures == [1]
    assert _jobs(rq)[0]["state"] == "done"


def test_failed_write_leaves_no_temp_file_and_old_ones_are_swept(
    rq: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = rq.Queue(rq.queue_dir())
    _submit(rq, "w", "pass")
    job = queue.jobs()[0]
    with monkeypatch.context() as patch:
        patch.setattr(rq.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("full")))
        with pytest.raises(OSError):
            queue.write(job)
    assert list(queue.jobs_dir.glob("*.tmp*")) == []
    old = queue.jobs_dir / "x.tmp999"
    old.write_text("partial")
    os.utime(old, (1, 1))
    rq.Queue(rq.queue_dir())
    assert not old.exists()


# --- corrupt and vanished records ------------------------------------------------------------


@pytest.mark.parametrize("content", ["{not json", "{}", '{"id": "x", "state": "queued"}'])
def test_unusable_record_is_moved_aside_with_a_note(
    rq: ModuleType, capsys: pytest.CaptureFixture, content: str
) -> None:
    _submit(rq, "good", "pass")
    bad = rq.queue_dir() / "jobs" / "bad.json"
    bad.write_text(content)
    capsys.readouterr()
    assert rq.main(["status"]) == 0
    captured = capsys.readouterr()
    assert "good" in captured.out and "bad.json" in captured.err
    assert not bad.exists() and (rq.queue_dir() / "jobs" / "bad.json.corrupt").exists()


def test_worker_survives_a_record_deleted_while_its_job_runs(rq: ModuleType) -> None:
    _submit(rq, "doomed", "import time; time.sleep(0.4)")
    _submit(rq, "other", "pass")
    worker = _start_worker(rq)
    try:
        deadline = time.time() + 10
        while not any(j["state"] == "running" for j in _jobs(rq)) and time.time() < deadline:
            time.sleep(0.02)
        (rq.queue_dir() / "jobs" / f"{_jobs(rq)[0]['id']}.json").unlink()
        assert worker.wait(timeout=15) == 0
    finally:
        worker.kill()
    assert any(j["name"] == "other" and j["state"] == "done" for j in _jobs(rq))


def test_wait_counts_a_vanished_record_as_a_failure(rq: ModuleType) -> None:
    _submit(rq, "vanish", "import time; time.sleep(0.6)", "v")
    worker = _start_worker(rq)
    job_path = rq.queue_dir() / "jobs" / f"{_jobs(rq)[0]['id']}.json"
    threading.Timer(0.3, lambda: job_path.unlink(missing_ok=True)).start()
    try:
        assert _bounded(lambda: rq.main(["wait", "--item", "v"])) == 1
    finally:
        worker.wait(timeout=15)
        worker.kill()


# --- prune -----------------------------------------------------------------------------------


def test_prune_moves_old_finished_records_aside(rq: ModuleType) -> None:
    _submit(rq, "old", "pass")
    _submit(rq, "fresh", "pass")
    _submit(rq, "queued", "pass")
    rq.main(["run"])
    queue = rq.Queue(rq.queue_dir())
    old = next(j for j in queue.jobs() if j["name"] == "old")
    old["end"] = time.time() - 30 * 86400
    queue.write(old)
    _submit(rq, "still-queued", "pass")
    assert rq.main(["prune", "--older-than", "7"]) == 0
    assert {j["name"] for j in _jobs(rq)} == {"fresh", "queued", "still-queued"}
    assert (queue.root / "done" / f"{old['id']}.json").exists()
    assert rq.main(["prune", "--older-than", "0"]) == 0
    assert [j["name"] for j in _jobs(rq)] == ["still-queued"]


# --- round 2: bad logs, record types, submitter environment, starting state, wrapper -------------


def _run_worker_bounded(rq: ModuleType) -> int:
    return _bounded(lambda: rq.main(["run"]))


def _edit(rq: ModuleType, name: str, **fields: object) -> dict:
    queue = rq.Queue(rq.queue_dir())
    job = next(j for j in queue.jobs() if j["name"] == name)
    job.update(fields)
    queue.write(job)
    return job


def test_job_with_an_unwritable_log_fails_and_does_not_block_the_queue(
    rq: ModuleType,
) -> None:
    _submit(rq, "badlog", "pass")
    _submit(rq, "fine", "pass")
    _edit(rq, "badlog", log="/proc/nope/x.log")
    assert _run_worker_bounded(rq) == 0
    states = {j["name"]: j for j in _jobs(rq)}
    assert states["badlog"]["state"] == "failed" and states["badlog"]["error"]
    assert states["fine"]["state"] == "done"


def test_submit_rejects_a_missing_cwd_and_an_uncreatable_log(
    rq: ModuleType, tmp_path: Path
) -> None:
    base = ["submit", "--name", "n", "--item", "i", "--no-worker"]
    assert rq.main([*base, "--cwd", str(tmp_path / "nope"), "--", "true"]) == 2
    assert rq.main([*base, "--log", "/proc/nope/x.log", "--", "true"]) == 2
    assert _jobs(rq) == []


def test_auto_started_worker_writes_its_stderr_to_worker_log(rq: ModuleType) -> None:
    queue = rq.Queue(rq.queue_dir())
    (queue.jobs_dir / "junk.json").write_text("{not json")
    rq.real_spawn_worker(queue)
    log = queue.root / "worker.log"
    deadline = time.time() + 10
    while not (log.exists() and "junk.json" in log.read_text()) and time.time() < deadline:
        time.sleep(0.05)
    assert "junk.json" in log.read_text()


@pytest.mark.parametrize(
    "field",
    [
        {"env": []},
        {"env": {"A": 1}},
        {"item": None},
        {"argv": "true"},
        {"state": "weird"},
        {"id": "some-other-id"},
        {"pid": "12"},
        {"name": 5},
    ],
    ids=lambda f: next(iter(f)) + "=" + repr(next(iter(f.values()))),
)
def test_record_with_wrong_types_is_quarantined(
    rq: ModuleType, capsys: pytest.CaptureFixture, field: dict
) -> None:
    _submit(rq, "good", "pass")
    queue = rq.Queue(rq.queue_dir())
    bad = {**queue.jobs()[0], "id": "bad", **field}
    (queue.jobs_dir / "bad.json").write_text(__import__("json").dumps(bad))
    capsys.readouterr()
    assert rq.main(["status"]) == 0
    captured = capsys.readouterr()
    assert "good" in captured.out and "bad.json" in captured.err
    assert (queue.jobs_dir / "bad.json.corrupt").exists()
    assert _run_worker_bounded(rq) == 0  # the worker is not killed by it either


def test_each_job_runs_with_its_own_submitters_environment(
    rq: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "env"
    code = (
        f"import os; open({str(out)!r} + os.environ['TAG'], 'w').write("
        "os.environ['FOO'] + '|' + os.environ['PYTHONPATH'] + '|' + os.environ['OMP_NUM_THREADS'] + '|' + os.environ.get('WORKERONLY', 'absent'))"
    )
    for tag in ("a", "b"):
        monkeypatch.setenv("FOO", f"foo-{tag}")
        monkeypatch.setenv("PYTHONPATH", f"/tree-{tag}")
        monkeypatch.setenv("TAG", tag)
        _submit(rq, tag, code)
    monkeypatch.setenv("FOO", "the-worker")  # the worker's own environment must not leak in
    monkeypatch.setenv("PYTHONPATH", "/worker-tree")
    monkeypatch.setenv("TAG", "worker")
    monkeypatch.setenv("WORKERONLY", "leak")
    assert _run_worker_bounded(rq) == 0
    assert (tmp_path / "enva").read_text() == "foo-a|/tree-a|1|absent"
    assert (tmp_path / "envb").read_text() == "foo-b|/tree-b|1|absent"


def test_starting_record_holds_a_slot_and_is_never_requeued_when_its_worker_died(
    rq: ModuleType,
) -> None:
    queue, job = _make_running(rq, worker=os.getpid(), worker_start=rq.proc_start(os.getpid()))
    job.update(state="starting", start=None)
    queue.write(job)
    assert queue.running_count() == 1
    with queue.locked():
        queue.reclaim_stale()
    assert queue.jobs()[0]["state"] == "starting"  # its worker is alive
    job.update(worker=_dead_pid(), worker_start=None)
    queue.write(job)
    with queue.locked():
        queue.reclaim_stale()
    got = queue.jobs()[0]
    assert got["state"] == "failed" and "lost" in got["error"]


def test_a_job_runs_at_most_once_and_keeps_its_slot_when_the_worker_dies_after_launching_it(
    rq: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RL_QUEUE_SLOTS", "1")
    marker, second = tmp_path / "ran.txt", tmp_path / "second.txt"
    first_code = (
        "import time; time.sleep(0.8); "
        f"open({str(marker)!r}, 'a').write(str(time.time()) + chr(10))"
    )
    _submit(rq, "once", first_code)
    _submit(rq, "next", f"import time; open({str(second)!r}, 'w').write(str(time.time()))")
    crasher = (
        "import importlib.util as u, os, sys\n"
        f"s = u.spec_from_file_location('rq', {str(SCRIPT)!r}); m = u.module_from_spec(s)\n"
        "sys.modules['rq'] = m; s.loader.exec_module(m)\n"
        "real = m.Queue.write\n"
        "def write(self, job):\n"
        "    if job['state'] == 'running': os._exit(9)\n"
        "    real(self, job)\n"
        "m.Queue.write = write\n"
        "m.main(['run'])\n"
    )
    assert subprocess.run([sys.executable, "-c", crasher], env=os.environ.copy()).returncode == 9
    first = _jobs(rq)[0]
    assert first["state"] == "starting"
    pid_file = rq.queue_dir() / "exit" / f"{first['id']}.pid"
    deadline = time.time() + 10
    while not pid_file.exists() and time.time() < deadline:
        time.sleep(0.02)  # the wrapper records itself as its first act
    assert pid_file.exists(), "the wrapper must record its pid, pgid and start time first"
    assert _run_worker_bounded(rq) == 0
    first, nxt = _jobs(rq)
    assert (first["state"], first["exit_code"]) == ("done", 0)  # its real result, not "lost"
    assert nxt["state"] == "done"
    assert len(marker.read_text().splitlines()) == 1  # ran once
    assert float(second.read_text()) >= float(marker.read_text())  # no overlap with the cap of 1


def test_starting_record_without_a_pid_file_is_given_time_to_launch(
    rq: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, job = _make_running(rq, worker=_dead_pid())
    job.update(state="starting", start=None, launching=time.time())
    queue.write(job)
    with queue.locked():
        queue.reclaim_stale()
    assert queue.jobs()[0]["state"] == "starting"  # the wrapper may not have started yet
    monkeypatch.setattr(rq, "LAUNCH_GRACE_SECONDS", 0.0)
    with queue.locked():
        queue.reclaim_stale()
    assert queue.jobs()[0]["state"] == "failed"


def _job_with_child(rq: ModuleType, tmp_path: Path, child_code: str = "") -> tuple[Path, Path]:
    pid_file = tmp_path / "child.pid"
    code = (
        f"import subprocess, sys, time; c = subprocess.Popen([sys.executable, '-c', {child_code!r}"
        f" or 'import time; time.sleep(60)']); open({str(pid_file)!r}, 'w').write(str(c.pid));"
        " c.wait()"
    )
    _submit(rq, "withchild", code)
    return pid_file, tmp_path


def test_killing_the_wrapper_keeps_the_slot_while_the_job_runs(
    rq: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    pid_file, _ = _job_with_child(rq, tmp_path)
    worker = _start_worker(rq)
    try:
        child = _wait_for(pid_file)
        job = _jobs(rq)[0]
        os.kill(job["pid"], signal.SIGKILL)
        time.sleep(0.5)  # several worker passes
        assert _alive(child)
        got = _jobs(rq)[0]
        assert got["state"] == "running" and rq.Queue(rq.queue_dir()).running_count() == 1
        capsys.readouterr()
        rq.main(["status"])
        assert "running(orphan)" in capsys.readouterr().out
        os.killpg(job["pgid"], signal.SIGKILL)
        assert worker.wait(timeout=15) == 0
    finally:
        worker.kill()
    final = _jobs(rq)[0]
    assert final["state"] == "failed" and "wrapper killed" in final["error"]


def test_wrapper_forwards_sigterm_to_its_child_and_exits_with_it(
    rq: ModuleType, tmp_path: Path
) -> None:
    pid_file = tmp_path / "job.pid"
    code = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    _submit(rq, "plain", code)
    worker = _start_worker(rq)
    try:
        child = _wait_for(pid_file)
        os.kill(_jobs(rq)[0]["pid"], signal.SIGTERM)  # the wrapper only, not the group
        deadline = time.time() + 5
        while _alive(child) and time.time() < deadline:
            time.sleep(0.05)
        assert not _alive(child)
        assert worker.wait(timeout=15) == 0
    finally:
        worker.kill()
    job = _jobs(rq)[0]
    assert job["state"] == "failed" and job["exit_code"] == -signal.SIGTERM


def test_status_shows_pgid_and_starts_a_worker_for_queued_jobs(
    rq: ModuleType, capsys: pytest.CaptureFixture
) -> None:
    _submit(rq, "q", "pass")
    capsys.readouterr()
    assert rq.main(["status"]) == 0
    assert rq.spawned == [1]
    assert "pgid=" in capsys.readouterr().out
    held = rq.try_worker_lock(rq.Queue(rq.queue_dir()))
    assert held is not None
    rq.main(["status"])
    assert rq.spawned == [1]  # a worker holds the lock: none is started
    rq.release_worker_lock(held)


def test_cancel_does_not_mark_a_job_cancelled_while_its_group_survives(
    rq: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rq, "signal_group", lambda pgid, sig: None)  # nothing can be killed
    monkeypatch.setattr(rq, "KILL_GRACE_SECONDS", 0.2)
    sleeper = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        _make_running(
            rq,
            pid=sleeper.pid,
            pgid=sleeper.pid,
            pid_start=rq.proc_start(sleeper.pid),
            worker=os.getpid(),
            worker_start=rq.proc_start(os.getpid()),
        )
        assert rq.main(["cancel", _jobs(rq)[0]["id"]]) == 1
        job = _jobs(rq)[0]
        assert job["state"] == "running" and job["cancelling"] is True
    finally:
        sleeper.kill()
        sleeper.wait()


def test_wait_finds_a_job_that_was_pruned_while_waiting(
    rq: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _submit(rq, "short", "import time; time.sleep(0.2)", "p")
    worker = _start_worker(rq)  # polls every 0.05 s (environment copied now)
    monkeypatch.setenv("RL_QUEUE_POLL", "1.0")  # the waiter looks only once a second
    result: list[int] = []
    waiter = threading.Thread(target=lambda: result.append(rq.main(["wait", "--item", "p"])))
    waiter.daemon = True
    waiter.start()
    try:
        deadline = time.time() + 10
        while _jobs(rq)[0]["state"] != "done" and time.time() < deadline:
            time.sleep(0.05)
        assert rq.main(["prune", "--older-than", "0"]) == 0
        waiter.join(timeout=10)
        assert result == [0]
    finally:
        worker.kill()


def test_orphaned_group_keeps_the_slot_after_both_pid_and_worker_are_gone(
    rq: ModuleType,
) -> None:
    survivor = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        queue, _ = _make_running(rq, pid=_dead_pid(), pgid=survivor.pid, worker=_dead_pid())
        with queue.locked():
            queue.reclaim_stale()
        assert queue.jobs()[0]["state"] == "running"
        survivor.kill()
        survivor.wait()
        with queue.locked():
            queue.reclaim_stale()
        assert queue.jobs()[0]["state"] == "failed"
    finally:
        survivor.kill()
        survivor.wait()


# --- round 3: secrets, identity of a group, versions, zombies ---------------------------------


def test_secret_looking_variables_are_not_stored_unless_passed_with_env(
    rq: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FOO_API_KEY", "key-value")
    monkeypatch.setenv("GH_TOKEN", "token-value")
    monkeypatch.setenv("MY_PASSWORD", "pw")
    monkeypatch.setenv("PLAIN", "kept")
    out = tmp_path / "seen"
    code = (
        f"import os; open({str(out)!r}, 'w').write(' '.join(os.environ.get(k, '-') for k in "
        "('FOO_API_' + 'KEY', 'GH_' + 'TOKEN', 'MY_' + 'PASSWORD', 'PLAIN', 'EXPLICIT_' + 'SECRET')))"
    )
    _submit(rq, "secrets", code, "t", "--env", "EXPLICIT_SECRET=on-purpose")
    record = (rq.queue_dir() / "jobs" / f"{_jobs(rq)[0]['id']}.json").read_text()
    for leaked in ("key-value", "token-value", "FOO_API_KEY", "GH_TOKEN", "MY_PASSWORD"):
        assert leaked not in record
    assert "on-purpose" in record
    assert _run_worker_bounded(rq) == 0
    assert out.read_text() == "- - - kept on-purpose"


def test_explicit_env_may_pass_a_secret_looking_name(rq: ModuleType) -> None:
    _submit(rq, "explicit", "pass", "t", "--env", "GH_TOKEN=abc")
    assert _jobs(rq)[0]["env"]["GH_TOKEN"] == "abc"


def test_record_is_created_private_and_the_queue_directories_are_owner_only(
    rq: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[tuple[str, int, int]] = []
    real_open = os.open

    def spy(path: str, flags: int, mode: int = 0o777, **kw: object) -> int:
        opened.append((str(path), flags, mode))
        return real_open(path, flags, mode, **kw)

    old_umask = os.umask(0o002)
    try:
        existing = tmp_path / "queue" / "logs"
        existing.mkdir(parents=True)
        existing.chmod(0o775)
        with monkeypatch.context() as patch:
            patch.setattr(rq.os, "open", spy)
            _submit(rq, "private", "pass")
    finally:
        os.umask(old_umask)
    tmps = [o for o in opened if ".tmp" in o[0] and o[0].endswith(tuple("0123456789"))]
    assert tmps and all(mode == 0o600 and flags & os.O_EXCL for _, flags, mode in tmps)
    root = rq.queue_dir()
    for d in (root, root / "jobs", root / "logs", root / "exit", root / "done"):
        assert d.stat().st_mode & 0o777 == 0o700, d
    assert (root / "jobs" / f"{_jobs(rq)[0]['id']}.json").stat().st_mode & 0o777 == 0o600


def test_pid_reused_as_a_group_id_does_not_hold_a_slot(rq: ModuleType) -> None:
    """The job's wrapper is gone; an unrelated process now has that pid, so its group is not ours."""
    bystander = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        queue, _ = _make_running(
            rq, pid=_dead_pid(), pgid=bystander.pid, pid_start="1", worker=_dead_pid()
        )
        with queue.locked():
            queue.reclaim_stale()
        assert queue.jobs()[0]["state"] == "failed"
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait()


def test_cancel_with_a_live_worker_ignores_a_group_that_is_not_the_jobs(
    rq: ModuleType,
) -> None:
    bystander = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        _make_running(
            rq,
            pid=bystander.pid,
            pgid=bystander.pid,
            pid_start="1",
            worker=os.getpid(),
            worker_start=rq.proc_start(os.getpid()),
        )
        rq.main(["cancel", _jobs(rq)[0]["id"]])
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait()


def test_a_group_of_only_zombies_is_not_alive(rq: ModuleType) -> None:
    corpse = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    try:
        deadline = time.time() + 10
        while rq.proc_state(corpse.pid) != "Z" and time.time() < deadline:
            time.sleep(0.02)  # exited, but not reaped by us yet
        assert rq.proc_state(corpse.pid) == "Z"
        assert not rq.group_alive(corpse.pid)
    finally:
        corpse.wait()


def test_record_with_an_unknown_version_is_skipped_not_quarantined(
    rq: ModuleType, capsys: pytest.CaptureFixture
) -> None:
    _submit(rq, "now", "pass")
    queue = rq.Queue(rq.queue_dir())
    assert queue.jobs()[0]["v"] == 1
    future = {**queue.jobs()[0], "id": "future", "v": 2, "name": "from-the-future"}
    path = queue.jobs_dir / "future.json"
    path.write_text(__import__("json").dumps(future))
    capsys.readouterr()
    assert rq.main(["status"]) == 0
    captured = capsys.readouterr()
    assert "from-the-future" not in captured.out and "now" in captured.out
    assert "version" in captured.err
    assert path.exists() and not (queue.jobs_dir / "future.json.corrupt").exists()


@pytest.mark.parametrize(
    "field",
    [
        {"argv": ["true", "a\x00b"]},
        {"env": {"A=B": "1"}},
        {"env": {"A": "x\x00y"}},
        {"env": {"": "1"}},
    ],
    ids=["nul-in-argv", "equals-in-key", "nul-in-value", "empty-key"],
)
def test_record_that_cannot_be_launched_is_quarantined(rq: ModuleType, field: dict) -> None:
    _submit(rq, "good", "pass")
    queue = rq.Queue(rq.queue_dir())
    bad = {**queue.jobs()[0], "id": "bad", **field}
    (queue.jobs_dir / "bad.json").write_text(__import__("json").dumps(bad))
    assert [j["name"] for j in queue.jobs()] == ["good"]
    assert (queue.jobs_dir / "bad.json.corrupt").exists()


def test_launch_errors_other_than_oserror_fail_the_job(
    rq: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _submit(rq, "boom", "pass")
    _submit(rq, "fine", "pass")
    real = subprocess.Popen

    def popen(cmd: list[str], *a: object, **kw: object):
        if "boom" in str(kw.get("env", {})) or kw.get("cwd") == "RAISE":
            raise ValueError("embedded null byte")
        return real(cmd, *a, **kw)

    _edit(rq, "boom", cwd="RAISE")
    monkeypatch.setattr(rq.subprocess, "Popen", popen)
    assert _run_worker_bounded(rq) == 0
    states = {j["name"]: j["state"] for j in _jobs(rq)}
    assert states == {"boom": "failed", "fine": "done"}
