"""Public CLI: resolve manifests, then delegate to the common execution APIs."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys


def parser():
    cli = argparse.ArgumentParser(description="VLN plugin discovery, execution and offline evaluation")
    commands = cli.add_subparsers(dest="command", required=True)
    benchmarks = commands.add_parser("benchmarks", help="benchmark catalog and integration status")
    benchmark_commands = benchmarks.add_subparsers(dest="benchmark_command", required=True)
    benchmark_commands.add_parser("list")
    benchmark_commands.add_parser("inspect").add_argument("id")
    smoke = benchmark_commands.add_parser("smoke", help="check assets or run one real environment episode without a model")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--check-only", action="store_true")
    smoke.add_argument("--output")
    smoke.add_argument("--plugin-dir", action="append", default=[])
    offline = benchmark_commands.add_parser("evaluate-offline", help="score recorded-data predictions")
    offline.add_argument("id")
    offline.add_argument("--reference", required=True)
    offline.add_argument("--predictions", required=True)
    offline.add_argument("--output")
    export = benchmark_commands.add_parser("export-scand", help="export a ROS1 odometry topic to JSONL")
    export.add_argument("--bag", required=True)
    export.add_argument("--topic", required=True)
    export.add_argument("--episode-id", required=True)
    export.add_argument("--output", required=True)
    plugins = commands.add_parser("plugins")
    plugin_commands = plugins.add_subparsers(dest="plugin_command", required=True)
    for name in ("list", "inspect", "check"):
        sub = plugin_commands.add_parser(name)
        if name != "list":
            sub.add_argument("id")
        sub.add_argument("--kind")
        sub.add_argument("--plugin-dir", action="append", default=[])
    plan = commands.add_parser("plan")
    plan.add_argument("--config", required=True)
    plan_gpus = plan.add_mutually_exclusive_group()
    plan_gpus.add_argument("--gpu", type=int, help="physical GPU for both roles this invocation")
    plan_gpus.add_argument("--gpus", help="comma-separated physical GPUs, one replica per GPU")
    plan.add_argument("--output")
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run_gpus = run.add_mutually_exclusive_group()
    run_gpus.add_argument("--gpu", type=int, help="physical GPU for both roles this invocation")
    run_gpus.add_argument("--gpus", help="comma-separated physical GPUs, one replica per GPU")
    run.add_argument("--output")
    configure = commands.add_parser("configure", help="register installed environments/assets once in configs/local.json")
    configure.add_argument("--plugin-dir", action="append", default=[], help="discover an external plugin bundle")
    configure.add_argument("--from", dest="source", help="import an existing installation map once")
    configure.add_argument("--method")
    configure.add_argument("--checkpoint")
    configure.add_argument("--method-python")
    configure.add_argument("--method-path", action="append", default=[], metavar="KEY=PATH")
    configure.add_argument("--simulator")
    configure.add_argument("--environment-python")
    configure.add_argument("--data-root")
    resume = commands.add_parser("resume")
    resume.add_argument("--run", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--metric-set")
    evaluate.add_argument("--plugin-dir", action="append", default=[])
    commands.add_parser("doctor")
    matrix = commands.add_parser("matrix")
    matrix.add_argument("--track", choices=("native", "standardized"), default="standardized")
    matrix.add_argument("--plugin-dir", action="append", default=[])
    compare = commands.add_parser("sim-compare")
    compare.add_argument("--runs", nargs="+", required=True)
    compare.add_argument("--output")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from nav_eval.plugins import Registry
        from nav_eval.storage import read_json, write_json
        if args.command == "benchmarks":
            from nav_eval.benchmarks import catalog, inspect_benchmark
            if args.benchmark_command == "list":
                result = catalog()
            elif args.benchmark_command == "inspect":
                result = inspect_benchmark(args.id)
            elif args.benchmark_command == "smoke":
                from nav_eval.benchmark_smoke import smoke_benchmark
                result = smoke_benchmark(read_json(args.config), check_only=args.check_only,
                                         registry=Registry(args.plugin_dir))
                if args.output:
                    write_json(args.output, result)
            elif args.benchmark_command == "export-scand":
                from extensions.benchmarks.socialnav.scand.evaluate import export_rosbag
                result = export_rosbag(args.bag, args.topic, args.episode_id, args.output)
            else:
                entry = inspect_benchmark(args.id)
                if entry["execution_mode"] != "offline" or entry["integration_status"] != "implemented":
                    raise ValueError(f"{entry['id']} has no offline recorded-data evaluator")
                result = Registry().get("benchmark", entry["id"]).load()(args.reference, args.predictions)
                if args.output:
                    write_json(args.output, result)
        elif args.command == "plugins":
            registry = Registry(args.plugin_dir)
            if args.plugin_command == "list":
                result = registry.list(args.kind)
            else:
                matches = [p for (kind, pid), p in registry.entries.items()
                           if pid == args.id and (args.kind is None or kind == args.kind)]
                if len(matches) != 1:
                    raise ValueError("plugin ID missing or ambiguous; specify --kind")
                plugin = matches[0]
                result = {**plugin.manifest, "identity": plugin.identity(), "location": str(plugin.root)}
                if args.plugin_command == "check":
                    entrypoint = plugin.manifest.get("entrypoint")
                    if entrypoint:
                        plugin.load()
                    elif plugin.manifest["kind"] == "benchmark":
                        for binding in plugin.manifest.get("bindings", {}).values():
                            plugin.load(binding["entrypoint"])
                    elif plugin.manifest["kind"] == "metric":
                        from nav_eval.evaluation.metrics import load_metric
                        for spec in plugin.manifest["metric_set"]["metrics"]:
                            load_metric(spec, plugin.root)
                    else:
                        raise ValueError(f"unbound implementation: {plugin.manifest['id']}")
                    result["check"] = "manifest_and_entrypoints_valid"
                    result["runtime_verified"] = False
        elif args.command == "configure":
            from nav_eval.local_config import configure
            result = configure(args)
        elif args.command in {"plan", "run"}:
            from nav_eval.planning import resolve
            from nav_eval.local_config import load_resources
            config = read_json(args.config)
            resources = load_resources(config)
            if args.gpus is not None:
                gpus = [int(value.strip()) for value in args.gpus.split(",")]
                if any(gpu < 0 for gpu in gpus) or len(set(gpus)) != len(gpus):
                    raise ValueError("--gpus needs unique nonnegative physical GPU indices")
                config["parallelism"] = len(gpus)
                resources["replicas"] = [{role: {"gpu": gpu} for role in ("method", "environment")}
                                         for gpu in gpus]
            if args.gpu is not None:
                if args.gpu < 0:
                    raise ValueError("--gpu must be a nonnegative physical GPU index")
                for role in ("method", "environment"):
                    resources.setdefault(role, {})["gpu"] = args.gpu
                resources["replicas"] = [{role: {"gpu": args.gpu} for role in ("method", "environment")}
                                         for _ in range(config.get("parallelism", 1))]
            if args.command == "plan":
                result = resolve(config, resources)
                if args.output:
                    write_json(args.output, result)
            else:
                from nav_eval.execution.runner import run_experiment
                result = run_experiment(config, resources, output=args.output)
        elif args.command == "resume":
            from nav_eval.execution.runner import run_experiment
            result = run_experiment(resume=args.run)
        elif args.command == "evaluate":
            from nav_eval.evaluation.runner import evaluate_run
            result = evaluate_run(args.run, args.metric_set, plugin_dirs=args.plugin_dir)
        elif args.command == "matrix":
            from nav_eval.planning import resolve
            registry = Registry(args.plugin_dir)
            result = []
            for benchmark in registry.list("benchmark"):
                for simulator in benchmark.get("bindings", {}):
                    for method in registry.list("method"):
                        config = {"method": method["id"], "benchmark": benchmark["id"],
                                  "simulator": simulator, "track": args.track}
                        try:
                            resolved = resolve(config, registry=registry)
                            status = ("compatibility_unknown" if resolved["compatibility"]["goals"] == "unknown"
                                      else "declared_compatible")
                            result.append({**config, "status": status, "runtime_verified": False})
                        except ValueError as error:
                            result.append({**config, "status": "incompatible", "reason": str(error)})
        elif args.command == "doctor":
            result = {"python": sys.version.split()[0], "platform": platform.platform(),
                      "docker": shutil.which("docker"), "nvidia_smi": shutil.which("nvidia-smi"),
                      "plugins": len(Registry().entries), "note": "Runtime and assets are checked by run preflight."}
        elif args.command == "sim-compare":
            from nav_eval.sim_compare import compare_runs
            result = compare_runs(args.runs, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if isinstance(result, dict) and result.get("status") in {"incomplete", "blocked"} else 0
    except (ValueError, KeyError, OSError, TimeoutError, ImportError, AttributeError, RuntimeError) as error:
        print(f"nav-eval: {error}", file=sys.stderr)
        return 2
