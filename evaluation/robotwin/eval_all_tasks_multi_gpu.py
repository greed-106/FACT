#!/usr/bin/env python3
"""Dynamically schedule RoboTwin-Phys tasks across persistent FACT GPU workers.

Each worker owns one FACT inference server and one RoboTwin-Phys rollout at a
time.  One or more workers can be assigned to each physical GPU. Completed
workers claim the next queued task from SQLite, so task duration differences do
not leave GPUs idle.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_SCRIPT = REPO_ROOT / "evaluation" / "robotwin" / "launch_server.sh"
CLIENT_SCRIPT = REPO_ROOT / "evaluation" / "robotwin" / "launch_client.sh"
DEFAULT_LAUNCH_CONFIG = REPO_ROOT / "evaluation" / "robotwin" / "launch_config.yml"
TASK_NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")


class SchedulerTerminated(Exception):
    """Raised by SIGTERM so running clients can be cleaned up first."""


@dataclass
class RunningTask:
    job_id: int
    task_name: str
    process: subprocess.Popen[Any]
    started_monotonic: float
    started_wall_time: float


@dataclass
class Worker:
    gpu_id: str
    slot: int
    port: int
    server: subprocess.Popen[Any]
    server_log: Path
    running: RunningTask | None = None


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_gpu_ids(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values or any(not item.isdecimal() for item in values):
        raise ValueError("--gpu-ids must be a non-empty comma-separated list of non-negative GPU indices.")
    gpu_ids = [str(int(item)) for item in values]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("--gpu-ids must not contain duplicates.")
    return gpu_ids


def _resolve_path(raw: str) -> Path:
    value = os.path.expanduser(os.path.expandvars(raw))
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _resolve_robotwin_path(cli_value: str | None, launch_config: Path) -> Path:
    raw = cli_value or os.environ.get("ROBOTWIN_PATH")
    if raw is None:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Pass --robotwin-path or install PyYAML to read the launch config.") from exc
        config = yaml.safe_load(launch_config.read_text(encoding="utf-8")) or {}
        raw = config.get("client", {}).get("ROBOTWIN_PATH")
    if not isinstance(raw, str) or not raw.strip() or raw.startswith("/your/path/to/"):
        raise ValueError("Set ROBOTWIN_PATH in launch_config.yml or pass --robotwin-path.")
    path = _resolve_path(raw)
    if not path.is_dir():
        raise FileNotFoundError(f"RoboTwin-Phys checkout not found: {path}")
    return path


def _load_tasks(robotwin_path: Path, task_list: str | None) -> list[str]:
    if task_list:
        tasks = [item.strip() for item in task_list.split(",") if item.strip()]
    else:
        candidates = (
            robotwin_path / "env_cfg" / "task_config" / "_eval_step_limit.yml",
            robotwin_path / "task_config" / "_eval_step_limit.yml",
        )
        step_limit = next((path for path in candidates if path.is_file()), None)
        if step_limit is None:
            raise FileNotFoundError(f"No _eval_step_limit.yml found below {robotwin_path}")
        tasks = re.findall(r"^([a-z0-9_]+):", step_limit.read_text(encoding="utf-8"), flags=re.MULTILINE)
    if not tasks or any(TASK_NAME_PATTERN.fullmatch(task) is None for task in tasks):
        raise ValueError("Task names must be non-empty lowercase letters, digits, or underscores.")
    if len(tasks) != len(set(tasks)):
        raise ValueError("Task list contains duplicates.")
    return tasks


def _connect(database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS experiments (
            name TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            config_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY,
            experiment_name TEXT NOT NULL REFERENCES experiments(name),
            task_name TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'completed', 'failed', 'timed_out', 'cancelled')),
            gpu_id TEXT,
            port INTEGER,
            started_at TEXT,
            finished_at TEXT,
            exit_code INTEGER,
            result_path TEXT,
            success INTEGER,
            total INTEGER,
            success_rate REAL,
            error_text TEXT,
            UNIQUE(experiment_name, task_name)
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            job_id INTEGER REFERENCES jobs(id),
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            message TEXT NOT NULL
        );
        """
    )
    return connection


def _event(connection: sqlite3.Connection, log_file: Path, job_id: int | None, event_type: str, message: str) -> None:
    timestamp = _now()
    connection.execute(
        "INSERT INTO events(job_id, timestamp, event_type, message) VALUES (?, ?, ?, ?)",
        (job_id, timestamp, event_type, message),
    )
    connection.commit()
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {event_type} {message}\n")


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _port_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def _server_environment(base: dict[str, str], *, gpu_id: str, port: int, robotwin_path: Path, launch_config: Path) -> dict[str, str]:
    environment = base.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "DEVICE": "cuda:0",
            "PORT": str(port),
            "SERVER_HOST": "127.0.0.1",
            "ROBOTWIN_PATH": str(robotwin_path),
            "ROBOTWIN_LAUNCH_CONFIG": str(launch_config),
        }
    )
    return environment


def _start_workers(
    *,
    gpu_ids: list[str],
    workers_per_gpu: int,
    base_port: int,
    base_environment: dict[str, str],
    robotwin_path: Path,
    launch_config: Path,
    worker_dir: Path,
    startup_timeout_seconds: float,
) -> list[Worker]:
    workers: list[Worker] = []
    for gpu_id in gpu_ids:
        for slot in range(workers_per_gpu):
            port = base_port + len(workers)
            server_log = worker_dir / f"gpu_{gpu_id}_slot_{slot}_server.log"
            with server_log.open("w", encoding="utf-8") as log_handle:
                server = subprocess.Popen(
                    ["bash", str(SERVER_SCRIPT)],
                    cwd=REPO_ROOT,
                    env=_server_environment(
                        base_environment,
                        gpu_id=gpu_id,
                        port=port,
                        robotwin_path=robotwin_path,
                        launch_config=launch_config,
                    ),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            workers.append(Worker(gpu_id=gpu_id, slot=slot, port=port, server=server, server_log=server_log))

    try:
        deadline = time.monotonic() + startup_timeout_seconds
        waiting = set(range(len(workers)))
        while waiting and time.monotonic() < deadline:
            for index in list(waiting):
                worker = workers[index]
                if worker.server.poll() is not None:
                    raise RuntimeError(
                        f"FACT server on GPU {worker.gpu_id} exited with {worker.server.returncode}; see {worker.server_log}"
                    )
                if _port_ready(worker.port):
                    waiting.remove(index)
            if waiting:
                time.sleep(1.0)
        if waiting:
            details = ", ".join(
                f"GPU {workers[index].gpu_id}, slot {workers[index].slot} ({workers[index].server_log})"
                for index in sorted(waiting)
            )
            raise TimeoutError(f"Timed out waiting for FACT servers: {details}")
    except BaseException:
        for worker in workers:
            _terminate(worker.server)
        raise
    return workers


def _find_result(robotwin_path: Path, task_name: str, started_wall_time: float) -> tuple[Path, int, int, float]:
    result_root = robotwin_path / "eval_result" / task_name
    if not result_root.is_dir():
        raise FileNotFoundError(f"No result directory created for {task_name}: {result_root}")
    candidates = [
        path
        for path in result_root.rglob("_result*.txt")
        if path.is_file() and path.stat().st_mtime >= started_wall_time
    ]
    if not candidates:
        raise FileNotFoundError(f"No result file written for {task_name} after its worker started")
    result_path = max(candidates, key=lambda path: path.stat().st_mtime)
    text = result_path.read_text(encoding="utf-8")
    rate: float | None = None
    attempts: int | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Attempts (rollouts + expert-infeasible):"):
            attempts = int(stripped.rsplit(":", 1)[1].strip())
        try:
            rate = float(stripped)
        except ValueError:
            pass
    if rate is None or not 0.0 <= rate <= 1.0:
        raise ValueError(f"Could not parse a valid success rate from {result_path}")
    if attempts is None or attempts <= 0:
        raise ValueError(f"Could not parse a positive attempt count from {result_path}")
    return result_path, int(round(rate * attempts)), attempts, rate


def _recover_interrupted_jobs(
    connection: sqlite3.Connection,
    *,
    experiment_name: str,
    robotwin_path: Path,
    scheduler_log: Path,
) -> tuple[int, int]:
    """Reconcile jobs left running when a scheduler controller disappears.

    A RoboTwin client writes its result before it exits.  Preserve those
    results, and return only jobs without a result to SQLite's queued state.
    This function must run after any orphaned client processes are stopped.
    """
    recovered = 0
    requeued = 0
    rows = connection.execute(
        "SELECT id, task_name, started_at FROM jobs WHERE experiment_name=? AND status='running' ORDER BY id",
        (experiment_name,),
    ).fetchall()
    for row in rows:
        task_name = str(row["task_name"])
        started_at = row["started_at"]
        try:
            started_wall_time = datetime.fromisoformat(str(started_at)).timestamp()
            result_path, success, total, rate = _find_result(robotwin_path, task_name, started_wall_time)
        except Exception:
            connection.execute(
                "UPDATE jobs SET status='queued', gpu_id=NULL, port=NULL, started_at=NULL, finished_at=NULL, "
                "exit_code=NULL, result_path=NULL, success=NULL, total=NULL, success_rate=NULL, error_text=NULL WHERE id=?",
                (row["id"],),
            )
            _event(connection, scheduler_log, int(row["id"]), "requeued", f"task={task_name} no result after interrupted scheduler")
            requeued += 1
        else:
            connection.execute(
                "UPDATE jobs SET status='completed', finished_at=?, exit_code=0, result_path=?, success=?, total=?, "
                "success_rate=?, error_text=NULL WHERE id=?",
                (_now(), str(result_path), success, total, rate, row["id"]),
            )
            _event(
                connection,
                scheduler_log,
                int(row["id"]),
                "recovered",
                f"task={task_name} success={success}/{total} rate={rate:.6f}",
            )
            recovered += 1
    connection.commit()
    return recovered, requeued


def _write_summary(connection: sqlite3.Connection, experiment_name: str, output_dir: Path) -> None:
    rows = connection.execute(
        "SELECT task_name, status, gpu_id, success, total, success_rate, result_path, error_text "
        "FROM jobs WHERE experiment_name=? ORDER BY id",
        (experiment_name,),
    ).fetchall()
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["task", "status", "gpu_id", "success", "total", "success_rate", "result_path", "error"])
        for row in rows:
            writer.writerow([row[key] for key in row.keys()])

    completed = [row for row in rows if row["status"] == "completed"]
    mean_rate = sum(float(row["success_rate"]) for row in completed) / len(completed) if completed else 0.0
    total_success = sum(int(row["success"]) for row in completed)
    total_attempts = sum(int(row["total"]) for row in completed)
    summary = {
        "experiment": experiment_name,
        "completed_tasks": len(completed),
        "total_tasks": len(rows),
        "mean_task_success_rate": mean_rate,
        "micro_success_rate": total_success / total_attempts if total_attempts else 0.0,
        "total_success": total_success,
        "total_attempts": total_attempts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"completed tasks: {summary['completed_tasks']}/{summary['total_tasks']}",
                f"mean task success rate: {summary['mean_task_success_rate']:.6f}",
                f"micro success rate: {summary['micro_success_rate']:.6f} ({total_success}/{total_attempts})",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _handle_sigterm(_signum: int, _frame: Any) -> None:
    raise SchedulerTerminated("scheduler received SIGTERM")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated physical GPU ids, for example 0,1,2,3.")
    parser.add_argument("--output-dir", required=True, help="New directory for SQLite state, logs, and summaries.")
    parser.add_argument("--experiment-name", default=None, help="SQLite experiment name; defaults to the output directory name.")
    parser.add_argument("--launch-config", default=str(DEFAULT_LAUNCH_CONFIG), help="FACT RoboTwin launcher YAML.")
    parser.add_argument("--robotwin-path", default=None, help="RoboTwin-Phys checkout; overrides ROBOTWIN_PATH and launch config.")
    parser.add_argument("--task-list", default=None, help="Optional comma-separated task subset; default is the benchmark's 50-task list.")
    parser.add_argument("--task-config", default="phys_random_all")
    parser.add_argument("--test-num", type=int, default=100, help="RoboTwin-Phys --eval_num_episodes per task.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Concurrent FACT server/client slots per physical GPU (default: 1).",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=18093,
        help="First local port; one consecutive port is assigned to each worker slot.",
    )
    parser.add_argument("--task-timeout-seconds", type=float, default=12 * 60 * 60)
    parser.add_argument("--server-startup-timeout-seconds", type=float, default=10 * 60)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true", help="Create and validate the SQLite queue without starting servers or clients.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Recover an interrupted run in --output-dir after its orphaned clients and servers have stopped.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    worker_count = len(gpu_ids) * args.workers_per_gpu
    if args.test_num <= 0 or args.workers_per_gpu <= 0 or args.base_port <= 0 or args.base_port + worker_count - 1 > 65535:
        raise ValueError("--test-num and --workers-per-gpu must be positive and the selected port range must be valid.")
    if args.task_timeout_seconds <= 0 or args.server_startup_timeout_seconds <= 0 or args.poll_interval_seconds <= 0:
        raise ValueError("Timeouts and poll interval must be positive.")
    launch_config = _resolve_path(args.launch_config)
    if not launch_config.is_file():
        raise FileNotFoundError(f"Launch config not found: {launch_config}")
    robotwin_path = _resolve_robotwin_path(args.robotwin_path, launch_config)
    if not (robotwin_path / "scripts" / "eval_policy.py").is_file():
        raise FileNotFoundError(f"Expected RoboTwin-Phys evaluator at {robotwin_path / 'scripts/eval_policy.py'}")
    tasks = _load_tasks(robotwin_path, args.task_list)

    output_dir = _resolve_path(args.output_dir)
    if args.resume:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Cannot resume missing output directory: {output_dir}")
    elif output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
        (output_dir / "jobs").mkdir()
    (output_dir / "jobs").mkdir(exist_ok=True)
    worker_dir = output_dir / "workers"
    worker_dir.mkdir(exist_ok=True)
    scheduler_log = output_dir / "scheduler.log"
    experiment_name = args.experiment_name or output_dir.name
    if not experiment_name:
        raise ValueError("--experiment-name is required when --output-dir has no final path component.")
    database = output_dir / "scheduler.sqlite3"
    run_config = {
        "experiment_name": experiment_name,
        "created_at": _now(),
        "launch_config": str(launch_config),
        "robotwin_path": str(robotwin_path),
        "gpu_ids": gpu_ids,
        "workers_per_gpu": args.workers_per_gpu,
        "ports": [args.base_port + index for index in range(worker_count)],
        "task_config": args.task_config,
        "test_num": args.test_num,
        "seed": args.seed,
        "tasks": tasks,
        "task_timeout_seconds": args.task_timeout_seconds,
    }
    if not args.resume:
        (output_dir / "scheduler-run-config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    connection = _connect(database)
    workers: list[Worker] = []
    result_code = 0
    previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        if args.resume:
            experiment = connection.execute("SELECT name FROM experiments WHERE name=?", (experiment_name,)).fetchone()
            if experiment is None:
                raise ValueError(f"No SQLite experiment named '{experiment_name}' in {database}")
            recovered, requeued = _recover_interrupted_jobs(
                connection,
                experiment_name=experiment_name,
                robotwin_path=robotwin_path,
                scheduler_log=scheduler_log,
            )
            _event(
                connection,
                scheduler_log,
                None,
                "scheduler_resumed",
                f"recovered={recovered} requeued={requeued} gpus={','.join(gpu_ids)} workers_per_gpu={args.workers_per_gpu}",
            )
        else:
            connection.execute(
                "INSERT INTO experiments(name, created_at, config_json) VALUES (?, ?, ?)",
                (experiment_name, _now(), json.dumps(run_config, ensure_ascii=False)),
            )
            connection.executemany(
                "INSERT INTO jobs(experiment_name, task_name, status) VALUES (?, ?, 'queued')",
                [(experiment_name, task_name) for task_name in tasks],
            )
            connection.commit()
            _event(
                connection,
                scheduler_log,
                None,
                "scheduler_started",
                f"tasks={len(tasks)} gpus={','.join(gpu_ids)} workers_per_gpu={args.workers_per_gpu}",
            )
        if args.dry_run:
            _event(connection, scheduler_log, None, "dry_run", "jobs queued without starting workers")
            _write_summary(connection, experiment_name, output_dir)
            return 0

        base_environment = os.environ.copy()
        workers = _start_workers(
            gpu_ids=gpu_ids,
            workers_per_gpu=args.workers_per_gpu,
            base_port=args.base_port,
            base_environment=base_environment,
            robotwin_path=robotwin_path,
            launch_config=launch_config,
            worker_dir=worker_dir,
            startup_timeout_seconds=args.server_startup_timeout_seconds,
        )
        for worker in workers:
            _event(
                connection,
                scheduler_log,
                None,
                "worker_ready",
                f"gpu={worker.gpu_id} slot={worker.slot} port={worker.port}",
            )

        def launch_next(worker: Worker) -> bool:
            row = connection.execute(
                "SELECT id, task_name FROM jobs WHERE experiment_name=? AND status='queued' ORDER BY id LIMIT 1",
                (experiment_name,),
            ).fetchone()
            if row is None:
                return False
            task_name = str(row["task_name"])
            task_log = output_dir / "jobs" / task_name / "client.log"
            task_log.parent.mkdir(parents=True, exist_ok=True)
            environment = _server_environment(
                base_environment,
                gpu_id=worker.gpu_id,
                port=worker.port,
                robotwin_path=robotwin_path,
                launch_config=launch_config,
            )
            environment["TEST_NUM"] = str(args.test_num)
            command = ["bash", str(CLIENT_SCRIPT), task_name, args.task_config, "", str(args.seed)]
            started_wall_time = time.time()
            with task_log.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            worker.running = RunningTask(
                job_id=int(row["id"]),
                task_name=task_name,
                process=process,
                started_monotonic=time.monotonic(),
                started_wall_time=started_wall_time,
            )
            connection.execute(
                "UPDATE jobs SET status='running', gpu_id=?, port=?, started_at=? WHERE id=?",
                (worker.gpu_id, worker.port, _now(), row["id"]),
            )
            _event(
                connection,
                scheduler_log,
                int(row["id"]),
                "started",
                f"task={task_name} gpu={worker.gpu_id} slot={worker.slot} port={worker.port}",
            )
            return True

        while True:
            for worker in workers:
                if worker.server.poll() is not None:
                    raise RuntimeError(
                        f"FACT server on GPU {worker.gpu_id} exited with {worker.server.returncode}; see {worker.server_log}"
                    )
                running = worker.running
                if running is None:
                    continue
                return_code = running.process.poll()
                elapsed = time.monotonic() - running.started_monotonic
                if return_code is None and elapsed < args.task_timeout_seconds:
                    continue
                if return_code is None:
                    _terminate(running.process)
                    connection.execute(
                        "UPDATE jobs SET status='timed_out', finished_at=?, exit_code=?, error_text=? WHERE id=?",
                        (_now(), running.process.returncode, f"task exceeded {args.task_timeout_seconds} seconds", running.job_id),
                    )
                    _event(
                        connection,
                        scheduler_log,
                        running.job_id,
                        "timed_out",
                        f"task={running.task_name} gpu={worker.gpu_id} slot={worker.slot}",
                    )
                elif return_code == 0:
                    try:
                        result_path, success, total, rate = _find_result(
                            robotwin_path, running.task_name, running.started_wall_time
                        )
                    except Exception as exc:
                        connection.execute(
                            "UPDATE jobs SET status='failed', finished_at=?, exit_code=?, error_text=? WHERE id=?",
                            (_now(), return_code, f"client exited successfully but result parsing failed: {exc}", running.job_id),
                        )
                        _event(connection, scheduler_log, running.job_id, "failed", f"task={running.task_name} result_parse={exc}")
                    else:
                        connection.execute(
                            "UPDATE jobs SET status='completed', finished_at=?, exit_code=?, result_path=?, success=?, total=?, success_rate=? WHERE id=?",
                            (_now(), return_code, str(result_path), success, total, rate, running.job_id),
                        )
                        _event(
                            connection,
                            scheduler_log,
                            running.job_id,
                            "completed",
                            f"task={running.task_name} gpu={worker.gpu_id} slot={worker.slot} success={success}/{total} rate={rate:.6f}",
                        )
                else:
                    connection.execute(
                        "UPDATE jobs SET status='failed', finished_at=?, exit_code=?, error_text=? WHERE id=?",
                        (_now(), return_code, f"client exited with {return_code}", running.job_id),
                    )
                    _event(
                        connection,
                        scheduler_log,
                        running.job_id,
                        "failed",
                        f"task={running.task_name} gpu={worker.gpu_id} slot={worker.slot} exit_code={return_code}",
                    )
                connection.commit()
                worker.running = None

            launched = False
            for worker in workers:
                if worker.running is None:
                    launched = launch_next(worker) or launched
            queued = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE experiment_name=? AND status='queued'", (experiment_name,)
            ).fetchone()[0]
            if queued == 0 and all(worker.running is None for worker in workers):
                break
            if not launched:
                time.sleep(args.poll_interval_seconds)
    except BaseException as exc:
        interrupted = isinstance(exc, (KeyboardInterrupt, SchedulerTerminated))
        reason = "scheduler interrupted" if interrupted else f"scheduler failed: {exc}"
        for worker in workers:
            if worker.running is None:
                continue
            try:
                _terminate(worker.running.process)
            finally:
                connection.execute(
                    "UPDATE jobs SET status='cancelled', finished_at=?, exit_code=?, error_text=? WHERE id=?",
                    (_now(), worker.running.process.returncode, reason, worker.running.job_id),
                )
                _event(connection, scheduler_log, worker.running.job_id, "cancelled", reason)
        connection.execute(
            "UPDATE jobs SET status='cancelled', finished_at=?, error_text=? WHERE experiment_name=? AND status='queued'",
            (_now(), reason, experiment_name),
        )
        _event(connection, scheduler_log, None, "scheduler_cancelled", reason)
        connection.commit()
        if not interrupted:
            raise
        result_code = 1
    finally:
        for worker in workers:
            _terminate(worker.server)
        _write_summary(connection, experiment_name, output_dir)
        if connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE experiment_name=? AND status!='completed'", (experiment_name,)
        ).fetchone()[0]:
            result_code = 1
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        connection.close()

    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
