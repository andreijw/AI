#!/usr/bin/env python3
"""A machine-wide job queue that caps how many training runs go at once (Linux).

Several agents share one small machine. The rule is at most 3 training or evaluation processes
at a time across all of them, each single-threaded. This queue enforces it, so nobody polls
``ps`` or waits for a go-ahead.

Usage (python standard library only):

    run_queue.py submit --name N --item ITEM [--cwd DIR] [--log FILE] [--env K=V ...] -- argv...
    run_queue.py run                       # worker: start queued jobs as slots free up
    run_queue.py status [--item ITEM]      # one line per job
    run_queue.py cancel ID | --item ITEM   # kill the job's process group, mark it cancelled
    run_queue.py wait ID | --item ITEM     # block until finished; exit 1 if any job did not succeed
    run_queue.py prune [--older-than DAYS] # move finished records older than DAYS (default 7) away

State is one JSON file per job in ``$RL_QUEUE_DIR`` (default ``~/rl-experiments/.queue``). The
cap is ``$RL_QUEUE_SLOTS`` (default 3); ``$RL_QUEUE_POLL`` (default 1 second) is how often a
worker or ``wait`` looks at the queue. Every change to the state happens under an exclusive
``flock`` on ``<queue>/lock``, so concurrent submitters and workers cannot overrun the cap.

Workers. There is at most one worker (``run``; ``worker`` is an alias): it holds
``flock(LOCK_EX | LOCK_NB)`` on ``<queue>/worker.lock`` for its whole life, so a second ``run``
exits at once. The kernel releases the lock when the worker dies, however it dies. ``submit``
(unless ``--no-worker``), ``wait`` and ``status`` start a detached worker, with its output in
``<queue>/worker.log``, whenever jobs are queued and nobody holds that lock. Before exiting, a
worker releases the lock and then looks at the queue once more; if a job arrived in that window,
it takes the lock again and carries on, so no job is stranded.

Jobs. A job runs in its own process group (its pgid is stored, shown by ``status``, and is the
only thing ``cancel`` signals), through a small wrapper that writes the job's real exit code to
``<queue>/exit/<id>`` and forwards SIGTERM and SIGINT to the job. A job gets exactly the
environment of the process that submitted it (stored in its record) plus ``--env``, with
OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and MKL_NUM_THREADS forced to 1; the worker's own
environment is not used. Variables whose names contain KEY, TOKEN, SECRET, PASSWORD, PASSWD,
CREDENTIAL, AUTH or COOKIE (any case) are not copied into the record, so a job does not see them
either; an explicit ``--env NAME=value`` passes one through on purpose. The queue directory is
owner-only (0700) and records are created 0600. ``submit`` exits 2 if ``--cwd`` is not a directory or the ``--log``
folder cannot be created; a job that still cannot start is marked ``failed`` with the error.
The argv is resolved with the submitter's PATH, so an absolute interpreter is safest.

States. A job is ``starting`` (written before launch, holds a slot, names the worker),
then ``running``. The wrapper's first act is to write ``exit/<id>.pid`` (its pid, pgid and start
time): if the worker dies right after launching, the next pass adopts the ``starting`` record as
``running`` from that file (a record with no pid file is given 5 seconds, then closed as lost).
Records carry ``"v": 1``; a record with another version is skipped with a note, not quarantined. A job holds its slot while its process, its worker or any process in its group
is alive (a group whose id is now the pid of an unrelated process does not count). A ``starting`` record whose worker died is closed as ``failed`` (error ``lost``) and
never re-queued, so a job runs at most once. A ``running`` record with no live process, worker
or group is closed from the exit file if the job finished, else ``failed`` (``lost``). Process
identity is checked by start time, so a reused pid is not mistaken for the job. A record that is
not valid JSON, has the wrong shape or types, or whose id differs from its file name is moved to
``<name>.corrupt`` with a note on stderr. A child that calls ``setsid`` leaves the group and
escapes ``cancel`` and the slot accounting.

Exit codes: 0 success; 1 ``wait`` found a job that failed, was cancelled or vanished, or
``cancel`` matched nothing or could not stop a group; 2 usage error (including ``wait`` with no
matching job).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import secrets
import signal
import subprocess  # nosec B404
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import IO, Any

try:
    import fcntl
except ImportError:  # Windows: the module imports, but the queue cannot run
    fcntl = None  # type: ignore[assignment]

DEFAULT_SLOTS = 3
DEFAULT_POLL_SECONDS = 1.0
KILL_GRACE_SECONDS = 3.0
STALE_TMP_SECONDS = 60.0
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
TERMINAL = ("done", "failed", "cancelled")
RECORD_VERSION = 1
LAUNCH_GRACE_SECONDS = 5.0
SECRET_NAME = re.compile(r"(?i)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE)")
STATES = ("queued", "starting", "running", *TERMINAL)
HOLDING_SLOT = ("starting", "running")
REQUIRED = ("id", "name", "item", "state", "cwd", "argv", "env", "log")

Job = dict[str, Any]

# Test hook: called after the worker released worker.lock and before it rechecks the queue.
before_idle_recheck: Callable[[], None] | None = None


class UsageError(Exception):
    """A bad argument or environment value; the command exits with code 2."""


def queue_dir() -> Path:
    """The queue directory: ``$RL_QUEUE_DIR`` or ``~/rl-experiments/.queue``."""
    override = os.environ.get("RL_QUEUE_DIR")
    return Path(override) if override else Path.home() / "rl-experiments" / ".queue"


def _env_number(name: str, default: float, cast: Callable[[str], float]) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = cast(raw)
    except ValueError:
        raise UsageError(f"{name}={raw!r} is not a number") from None
    if value <= 0:
        raise UsageError(f"{name} must be positive, got {raw!r}")
    return value


def slot_cap() -> int:
    """The most jobs that may run at once: ``$RL_QUEUE_SLOTS`` or 3."""
    return int(_env_number("RL_QUEUE_SLOTS", DEFAULT_SLOTS, int))


def poll_seconds() -> float:
    """How often a worker or ``wait`` looks at the queue: ``$RL_QUEUE_POLL`` or 1 second."""
    return _env_number("RL_QUEUE_POLL", DEFAULT_POLL_SECONDS, float)


def proc_start(pid: int | None) -> str | None:
    """The process start time (field 22 of /proc/<pid>/stat), or None if unavailable."""
    if not pid:
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def pid_alive(pid: int | None) -> bool:
    """Whether a live process has this pid (a zombie, which has exited, does not count)."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return proc_state(pid) != "Z"


def proc_state(pid: int) -> str | None:
    """The process state letter from /proc/<pid>/stat (``Z`` is a zombie), or None."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return stat.rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return None


def pid_matches(pid: int | None, start: str | None) -> bool:
    """Whether ``pid`` is alive and, when a start time was stored, is the same process."""
    if not pid_alive(pid):
        return False
    now = proc_start(pid)
    return start is None or now is None or now == start


def group_alive(pgid: int | None) -> bool:
    """Whether a live (non-zombie) process is left in this process group."""
    if not pgid:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # members we may not signal: fall through to the scan
    try:
        entries = [p for p in Path("/proc").iterdir() if p.name.isdigit()]
    except OSError:
        return True
    for entry in entries:
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                return True
        except (OSError, IndexError, ValueError):
            continue
    return False


def signal_group(pgid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def job_group_alive(job: Job) -> bool:
    """Whether the job's own process group is alive.

    False when a live process now has the group id as its pid but a different start time than
    the job's wrapper had: the id was reused by something unrelated.
    """
    pgid = job.get("pgid")
    start = job.get("pid_start")
    if pgid and start is not None and pid_alive(pgid):
        now = proc_start(pgid)
        if now is not None and now != start:
            return False
    return group_alive(pgid)


def _is_int(value: object) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool))


def _is_number(value: object) -> bool:
    return value is None or (isinstance(value, int | float) and not isinstance(value, bool))


def _problem(data: object, stem: str) -> str | None:
    """Why a parsed record is unusable (wrong shape, types or id), or None if it is fine."""
    if not isinstance(data, dict):
        return "not a JSON object"
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        return f"missing {', '.join(missing)}"
    strings = ("id", "name", "item", "state", "cwd", "log")
    if not all(isinstance(data[key], str) for key in strings):
        return "id, name, item, state, cwd and log must be strings"
    if data["id"] != stem:
        return "id does not match the file name"
    if data["state"] not in STATES:
        return f"unknown state {data['state']!r}"
    argv, env = data["argv"], data["env"]
    if not (isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv)):
        return "argv must be a non-empty list of strings"
    if any("\x00" in a for a in argv):
        return "argv contains a NUL byte"
    if not (
        isinstance(env, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
    ):
        return "env must map strings to strings"
    if any(not k or "=" in k or "\x00" in k or "\x00" in v for k, v in env.items()):
        return "env has an empty name, a name containing '=', or a NUL byte"
    if not all(_is_int(data.get(k)) for k in ("pid", "pgid", "worker", "exit_code")):
        return "pid, pgid, worker and exit_code must be integers or null"
    if not all(_is_number(data.get(k)) for k in ("submitted", "start", "end")):
        return "times must be numbers or null"
    return None


class Queue:
    """The on-disk queue: job records, exit files and the locks."""

    def __init__(self, root: Path) -> None:
        """Create the directory layout under ``root`` and sweep old temp files."""
        self.root = root
        self.jobs_dir = root / "jobs"
        self.logs_dir = root / "logs"
        self.exit_dir = root / "exit"
        self.done_dir = root / "done"
        for d in (root, self.jobs_dir, self.logs_dir, self.exit_dir, self.done_dir):
            d.mkdir(mode=0o700, parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                d.chmod(0o700)  # records hold environments; keep the queue owner-only
        for tmp in self.jobs_dir.glob("*.tmp*"):
            with contextlib.suppress(OSError):
                if time.time() - tmp.stat().st_mtime > STALE_TMP_SECONDS:
                    tmp.unlink()

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the queue's exclusive file lock."""
        if fcntl is None:
            raise SystemExit("run_queue.py needs fcntl (Linux or macOS)")
        with open(self.root / "lock", "a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def write(self, job: Job) -> None:
        """Write a job record atomically (temp file, then rename); no temp file is left behind."""
        path = self.jobs_dir / f"{job['id']}.json"
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        try:
            tmp.unlink(missing_ok=True)
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # private from creation
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(job, indent=2, sort_keys=True))
            os.replace(tmp, path)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise

    def _quarantine(self, path: Path, why: str) -> None:
        print(f"run_queue: {path.name} is unusable ({why}); moved to .corrupt", file=sys.stderr)
        with contextlib.suppress(OSError):
            os.replace(path, path.with_name(path.name + ".corrupt"))

    def jobs(self) -> list[Job]:
        """Every valid job record, oldest first. Unusable files are moved aside."""
        out: list[Job] = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as exc:
                self._quarantine(path, type(exc).__name__)
                continue
            if isinstance(data, dict) and data.get("v", RECORD_VERSION) != RECORD_VERSION:
                print(
                    f"run_queue: {path.name} has record version {data.get('v')!r}; skipped",
                    file=sys.stderr,
                )
                continue
            problem = _problem(data, path.stem)
            if problem:
                self._quarantine(path, problem)
                continue
            out.append(data)
        return out

    def select(self, job_id: str | None, item: str | None) -> list[Job]:
        """Jobs by id (or unique id prefix) or by item label."""
        jobs = self.jobs()
        if job_id:
            hits = [j for j in jobs if j["id"] == job_id] or [
                j for j in jobs if j["id"].startswith(job_id)
            ]
            if len(hits) > 1:
                raise UsageError(f"ambiguous job id {job_id!r}")
            return hits
        return [j for j in jobs if j["item"] == item]

    def archived(self, job_id: str) -> Job | None:
        """A record that ``prune`` moved to done/, or None."""
        try:
            data = json.loads((self.done_dir / f"{job_id}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if _problem(data, job_id) is None else None

    def read_exit(self, job_id: str) -> int | None:
        """The exit code the job's wrapper recorded, or None."""
        try:
            return int((self.exit_dir / job_id).read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def read_pidfile(self, job_id: str) -> dict[str, Any] | None:
        """What the wrapper recorded about itself (pid, pgid, start time), or None."""
        try:
            data = json.loads((self.exit_dir / f"{job_id}.pid").read_text(encoding="utf-8"))
            return data if _is_int(data.get("pid")) and _is_int(data.get("pgid")) else None
        except (OSError, ValueError, AttributeError):
            return None

    def reclaim_stale(self) -> None:
        """Close or adopt jobs whose worker is gone. Hold the lock.

        A ``running`` record stays open while its process, its worker or its process group is
        alive. A ``starting`` record whose worker is gone is adopted as ``running`` if the
        wrapper's pid file names a live process (the worker died just after launching it); with
        no pid file it is given ``LAUNCH_GRACE_SECONDS`` and then closed as ``failed``/``lost``.
        Nothing is ever re-queued, so a job runs at most once.
        """
        for job in self.jobs():
            if job["state"] not in HOLDING_SLOT:
                continue
            if pid_matches(job.get("worker"), job.get("worker_start")):
                continue
            if job["state"] == "starting":
                pidfile = self.read_pidfile(job["id"])
                if pidfile is None:
                    if time.time() - (job.get("launching") or 0) < LAUNCH_GRACE_SECONDS:
                        continue  # the wrapper may not have started yet
                else:
                    seen = {**job, "pid": pidfile["pid"], "pgid": pidfile["pgid"]}
                    seen["pid_start"] = pidfile.get("start")
                    if pid_matches(seen["pid"], seen["pid_start"]) or job_group_alive(seen):
                        job.update(
                            state="running",
                            pid=seen["pid"],
                            pgid=seen["pgid"],
                            pid_start=seen["pid_start"],
                            start=job.get("launching"),
                        )
                        self.write(job)
                        continue
            elif job["state"] == "running":
                if pid_matches(job.get("pid"), job.get("pid_start")):
                    continue
                if job_group_alive(job):
                    continue  # the wrapper is gone but the job is not: the slot stays taken
            code = self.read_exit(job["id"])
            if job.get("cancelling"):
                job.update(state="cancelled", end=time.time())
            elif code is None:
                job.update(state="failed", end=time.time(), error="lost: worker and job gone")
            else:
                job.update(state="done" if code == 0 else "failed", exit_code=code, end=time.time())
            self.write(job)

    def running_count(self) -> int:
        """How many jobs hold a slot. Hold the lock."""
        return sum(1 for j in self.jobs() if j["state"] in HOLDING_SLOT)


def try_worker_lock(queue: Queue) -> IO[bytes] | None:
    """Take ``worker.lock`` without blocking; None if another process holds it."""
    if fcntl is None:
        raise SystemExit("run_queue.py needs fcntl (Linux or macOS)")
    handle = open(queue.root / "worker.lock", "a+b")  # noqa: SIM115 - held by the caller
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def release_worker_lock(handle: IO[bytes]) -> None:
    """Release and close a lock taken by :func:`try_worker_lock`."""
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_UN)
    handle.close()


def spawn_worker(queue: Queue) -> None:
    """Start a detached worker for this queue; its output goes to ``<queue>/worker.log``."""
    with open(queue.root / "worker.log", "ab") as log:
        subprocess.Popen(  # nosec B603
            [sys.executable, os.path.abspath(__file__), "run"],
            env={**os.environ, "RL_QUEUE_DIR": str(queue.root)},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )


def ensure_worker(queue: Queue) -> bool:
    """Start a worker if nobody holds ``worker.lock``; True if one was started."""
    lock = try_worker_lock(queue)
    if lock is None:
        return False
    release_worker_lock(lock)
    spawn_worker(queue)
    return True


def submit(queue: Queue, args: argparse.Namespace) -> int:
    """Queue a job and print its id."""
    argv: list[str] = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        raise UsageError("submit: no command given after --")
    # The job gets exactly the submitter's environment (plus --env), whichever process later
    # starts the worker; otherwise one agent's PYTHONPATH could leak into another's job.
    # Variables that look like secrets are not copied into the record (it is a file on disk);
    # an explicit --env passes one through on purpose.
    env = {k: v for k, v in os.environ.items() if not SECRET_NAME.search(k)}
    for pair in args.env:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise UsageError(f"--env expects K=V, got {pair!r}")
        env[key] = value
    cwd = Path(args.cwd).resolve()
    if not cwd.is_dir():
        raise UsageError(f"--cwd {args.cwd!r} is not a directory")
    # Zero-padded nanoseconds sort in submission order, so the oldest queued job starts first.
    job_id = f"{time.time_ns():020d}-{secrets.token_hex(2)}"
    log = Path(args.log).resolve() if args.log else queue.logs_dir / f"{job_id}.log"
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise UsageError(f"cannot create the folder for --log {str(log)!r}: {exc}") from None
    job: Job = {
        "v": RECORD_VERSION,
        "id": job_id,
        "name": args.name,
        "item": args.item,
        "cwd": str(cwd),
        "argv": argv,
        "env": env,
        "state": "queued",
        "pid": None,
        "pgid": None,
        "pid_start": None,
        "worker": None,
        "worker_start": None,
        "submitted": time.time(),
        "start": None,
        "end": None,
        "exit_code": None,
        "log": str(log),
    }
    with queue.locked():
        queue.write(job)
    print(job_id)
    if not args.no_worker:
        ensure_worker(queue)  # after the write: see the module docstring on the lost wakeup
    return 0


def exec_wrapper(exit_file: str, argv: Sequence[str]) -> int:
    """Run ``argv`` and record its real exit code in ``exit_file`` (internal ``_exec``).

    SIGTERM and SIGINT are forwarded to the child, and the wrapper exits only when the child
    does, so signalling the wrapper alone cannot orphan a running job.
    """
    received: list[int] = []
    child: subprocess.Popen[bytes] | None = None
    # First act: say who we are, so a worker that dies right after launching us can be adopted.
    info = {"pid": os.getpid(), "pgid": os.getpgrp(), "start": proc_start(os.getpid())}
    pid_tmp = f"{exit_file}.pid.tmp{os.getpid()}"
    Path(pid_tmp).write_text(json.dumps(info), encoding="utf-8")
    os.replace(pid_tmp, f"{exit_file}.pid")

    def forward(signum: int, _frame: object) -> None:
        received.append(signum)
        if child is not None:
            with contextlib.suppress(OSError):
                child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        child = subprocess.Popen(list(argv))  # nosec B603
        for signum in received:  # a signal that arrived before the child existed
            with contextlib.suppress(OSError):
                child.send_signal(signum)
        code = child.wait()
    except OSError as exc:
        print(f"run_queue: cannot start {argv[0]!r}: {exc}", file=sys.stderr)
        code = 127
    tmp = f"{exit_file}.tmp{os.getpid()}"
    Path(tmp).write_text(str(code), encoding="utf-8")
    os.replace(tmp, exit_file)
    return code if code >= 0 else 128 - code


def _start(queue: Queue, job: Job) -> subprocess.Popen[bytes] | None:
    """Launch a queued job in its own process group and record it as running.

    The record is written as ``starting`` (with this worker's pid) before anything is launched,
    so a worker that dies after the launch cannot cause a second run: see ``reclaim_stale``.
    """
    env = {**job["env"], **dict.fromkeys(THREAD_VARS, "1")}
    log = Path(job["log"])
    exit_file = queue.exit_dir / job["id"]
    job.update(
        state="starting",
        worker=os.getpid(),
        worker_start=proc_start(os.getpid()),
        launching=time.time(),
    )
    queue.write(job)  # if this fails, nothing was launched and the job is still queued
    wrapper = [sys.executable, os.path.abspath(__file__), "_exec", str(exit_file), "--"]
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        exit_file.unlink(missing_ok=True)
        Path(f"{exit_file}.pid").unlink(missing_ok=True)
        with open(log, "ab") as out:
            proc = subprocess.Popen(  # nosec B603
                [*wrapper, *job["argv"]],
                cwd=job["cwd"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except (OSError, ValueError) as exc:
        job.update(state="failed", exit_code=127, end=time.time(), error=str(exc))
        queue.write(job)
        return None
    job.update(
        state="running",
        pid=proc.pid,
        pgid=proc.pid,
        pid_start=proc_start(proc.pid),
        start=time.time(),
    )
    try:
        queue.write(job)
    except OSError as exc:
        # Do not leave the job running unrecorded. If putting it back to queued fails too, the
        # record stays ``starting`` under this worker and is closed as lost, never run twice.
        signal_group(proc.pid, signal.SIGKILL)
        proc.wait()
        print(f"run_queue: could not record job {job['id']} as running: {exc}", file=sys.stderr)
        job.update(
            state="queued", pid=None, pgid=None, pid_start=None, worker=None, worker_start=None
        )
        job["start"] = None
        queue.write(job)
        return None
    return proc


def _reap(queue: Queue, children: dict[str, subprocess.Popen[bytes]]) -> None:
    """Record the exit of this worker's finished children. Hold the lock."""
    by_id = {j["id"]: j for j in queue.jobs()}
    for job_id, proc in list(children.items()):
        returncode = proc.poll()
        if returncode is None:
            continue
        job = by_id.get(job_id)
        if job is not None and job_group_alive(job):
            continue  # the wrapper is gone but the job is not: the slot stays taken
        if job is not None:
            recorded = queue.read_exit(job_id)
            code = returncode if recorded is None else recorded
            if recorded is None and returncode < 0:
                job["error"] = f"wrapper killed (signal {-returncode}); job exit unknown"
            if job.get("cancelling") or job["state"] == "cancelled":
                job.update(state="cancelled", end=job["end"] or time.time())
            else:
                state = "done" if code == 0 else "failed"
                job.update(state=state, exit_code=code, end=time.time())
            queue.write(job)  # may raise OSError: the child stays tracked and is retried
        del children[job_id]


def _tick(queue: Queue, children: dict[str, subprocess.Popen[bytes]], cap: int) -> bool:
    """One worker pass under the queue lock; True while there is work this worker must watch."""
    with queue.locked():
        _reap(queue, children)
        queue.reclaim_stale()
        free = max(0, cap - queue.running_count())
        queued = [j for j in queue.jobs() if j["state"] == "queued"]
        for job in queued[:free]:
            proc = _start(queue, job)
            if proc is not None:
                children[job["id"]] = proc
    return bool(children) or len(queued) > free


def run_worker(queue: Queue) -> int:
    """Start queued jobs as slots free up until the queue is empty and this worker's are done."""
    cap = slot_cap()
    poll = poll_seconds()
    lock = try_worker_lock(queue)
    if lock is None:
        return 0  # another worker is running; it will see everything queued
    children: dict[str, subprocess.Popen[bytes]] = {}
    try:
        while True:
            try:
                busy = _tick(queue, children, cap)
            except OSError as exc:
                print(f"run_queue: {exc}; retrying", file=sys.stderr)
                busy = True
            if busy:
                time.sleep(poll)
                continue
            # Idle. Release the lock, then look once more: a submitter that saw the lock held
            # started no worker, so anything it queued before now is ours.
            release_worker_lock(lock)
            lock = None
            if before_idle_recheck is not None:
                before_idle_recheck()
            with queue.locked():
                more = any(j["state"] == "queued" for j in queue.jobs())
            if not more:
                return 0
            lock = try_worker_lock(queue)
            if lock is None:
                return 0  # a new worker took over
    finally:
        if lock is not None:
            release_worker_lock(lock)


def _line(job: Job) -> str:
    """One status line for a job."""
    start, end = job.get("start"), job.get("end")
    seconds = ((end or time.time()) - start) if start else 0.0
    code = "-" if job.get("exit_code") is None else str(job["exit_code"])
    state = job["state"] + ("*" if job.get("cancelling") and job["state"] == "running" else "")
    wrapper_gone = not pid_matches(job.get("pid"), job.get("pid_start"))
    if job["state"] == "running" and wrapper_gone and job_group_alive(job):
        state = state.replace("running", "running(orphan)")
    return (
        f"{job['id']}  {state:<15} {job['item']:<16} {job['name']:<24} "
        f"pid={job.get('pid') or '-'} pgid={job.get('pgid') or '-'} exit={code} "
        f"{seconds:6.0f}s log={job['log']}"
    )


def status(queue: Queue, args: argparse.Namespace) -> int:
    """Print one line per job (reclaiming stale records first)."""
    with queue.locked():
        queue.reclaim_stale()
    for job in queue.jobs():
        if args.item is None or job["item"] == args.item:
            print(_line(job))
    if any(j["state"] == "queued" for j in queue.jobs()):
        ensure_worker(queue)  # queued jobs and nobody to run them
    return 0


def _terminate(job: Job) -> bool:
    """Stop the job's process group; True once it is gone (or was never this job's)."""
    pgid = job.get("pgid")
    if not pgid or not job_group_alive(job):
        return True  # gone, or the group id now belongs to something unrelated
    signal_group(pgid, signal.SIGTERM)
    for sig in (None, signal.SIGKILL):
        if sig is not None:
            signal_group(pgid, sig)
        deadline = time.time() + KILL_GRACE_SECONDS
        while group_alive(pgid) and time.time() < deadline:
            time.sleep(0.05)
        if not group_alive(pgid):
            return True
    return False


def cancel(queue: Queue, args: argparse.Namespace) -> int:
    """Cancel jobs. Queued ones are marked at once. A running job keeps its slot, flagged
    ``cancelling``, and is marked cancelled only once its process group is gone (signalled by
    stored pgid, never by pattern). A child that calls ``setsid`` leaves the group and escapes.
    Exit 1 if nothing matched or a group would not die.
    """
    targets: list[Job] = []
    with queue.locked():
        queue.reclaim_stale()
        selected = queue.select(args.id, args.item)
        if not selected:
            print("cancel: no matching job", file=sys.stderr)
            return 1
        for job in selected:
            if job["state"] == "queued":
                job.update(state="cancelled", end=time.time())
                queue.write(job)
                print(f"cancelled {job['id']} {job['name']}")
            elif job["state"] in HOLDING_SLOT:
                job["cancelling"] = True
                queue.write(job)
                targets.append(job)
    survivors = 0
    gone = {job["id"]: _terminate(job) for job in targets}
    if targets:
        with queue.locked():
            current = {j["id"]: j for j in queue.jobs()}
            for job in targets:
                now = current.get(job["id"])
                if not gone[job["id"]]:
                    survivors += 1
                    print(f"still running: {job['id']} {job['name']}", file=sys.stderr)
                elif now is not None and now["state"] in HOLDING_SLOT:
                    now.update(state="cancelled", end=time.time())
                    queue.write(now)
                    print(f"cancelled {job['id']} {job['name']}")
    return 1 if survivors else 0


def wait(queue: Queue, args: argparse.Namespace) -> int:
    """Block until the selected jobs finish; 1 if any failed, was cancelled or vanished."""
    ids = [j["id"] for j in queue.select(args.id, args.item)]
    if not ids:
        print("wait: no matching job", file=sys.stderr)
        return 2
    poll = poll_seconds()
    last_spawn = 0.0
    while True:
        with queue.locked():
            queue.reclaim_stale()
        by_id = {j["id"]: j for j in queue.jobs()}
        for job_id in ids:
            if job_id not in by_id and (archived := queue.archived(job_id)) is not None:
                by_id[job_id] = archived  # pruned while we waited
        jobs = [by_id[i] for i in ids if i in by_id]
        if all(j["state"] in TERMINAL for j in jobs):
            break
        queued = any(j["state"] == "queued" for j in jobs)
        if queued and time.time() - last_spawn > 5 and ensure_worker(queue):
            last_spawn = time.time()
        time.sleep(poll)
    bad = 0
    for job_id in ids:
        job = by_id.get(job_id)
        if job is None:
            print(f"vanished: {job_id}", file=sys.stderr)
            bad += 1
        elif job["state"] != "done":
            msg = f"{job['state']}: {job_id} {job['name']} exit={job.get('exit_code')}"
            print(msg, file=sys.stderr)
            bad += 1
    return 1 if bad else 0


def prune(queue: Queue, args: argparse.Namespace) -> int:
    """Move finished records (and their exit files) older than ``--older-than`` days to done/."""
    cutoff = time.time() - args.older_than * 86400
    moved = 0
    with queue.locked():
        for job in queue.jobs():
            if job["state"] in TERMINAL and (job.get("end") or 0) <= cutoff:
                os.replace(
                    queue.jobs_dir / f"{job['id']}.json", queue.done_dir / f"{job['id']}.json"
                )
                (queue.exit_dir / job["id"]).unlink(missing_ok=True)
                (queue.exit_dir / f"{job['id']}.pid").unlink(missing_ok=True)
                moved += 1
    print(f"pruned {moved} job(s) to {queue.done_dir}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("submit", help="queue a job (command after --)")
    p.add_argument("--name", required=True)
    p.add_argument("--item", required=True, help="owner label, e.g. the roadmap item")
    p.add_argument("--cwd", default=".")
    p.add_argument("--log", help="log file (default: <queue>/logs/<id>.log)")
    p.add_argument("--env", action="append", default=[], metavar="K=V")
    p.add_argument("--no-worker", action="store_true", help="do not start a worker")
    p.add_argument("argv", nargs=argparse.REMAINDER)
    for name in ("run", "worker"):
        sub.add_parser(name, help="start queued jobs as slots free up, until the queue is empty")
    p = sub.add_parser("status", help="one line per job")
    p.add_argument("--item")
    for name in ("cancel", "wait"):
        p = sub.add_parser(name)
        p.add_argument("id", nargs="?")
        p.add_argument("--item")
    p = sub.add_parser("prune", help="move finished records older than DAYS to done/")
    p.add_argument("--older-than", type=float, default=7.0, metavar="DAYS")
    p = sub.add_parser("_exec")  # internal: the per-job wrapper
    p.add_argument("exit_file")
    p.add_argument("argv", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the exit code."""
    args = _parser().parse_args(argv)
    if args.command == "_exec":
        command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        return exec_wrapper(args.exit_file, command)
    try:
        if args.command in ("cancel", "wait") and bool(args.id) == bool(args.item):
            raise UsageError(f"{args.command}: give exactly one of ID or --item")
        queue = Queue(queue_dir())
        handlers: dict[str, Callable[[Queue, argparse.Namespace], int]] = {
            "submit": submit,
            "run": lambda q, _: run_worker(q),
            "worker": lambda q, _: run_worker(q),
            "status": status,
            "cancel": cancel,
            "wait": wait,
            "prune": prune,
        }
        return handlers[args.command](queue, args)
    except UsageError as exc:
        print(f"run_queue: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
