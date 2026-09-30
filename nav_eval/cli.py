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
    plan.add_argument("--resources")
    plan.add_argument("--output")
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--resources")
    run.add_argument("--output")
    resume = commands.add_parser("resume")
    resume.add_argument("--run", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--metric-set")
    evaluate.add_argument("--plugin-dir", action="append", default=[])
    commands.add_parser("doctor")
    commands.add_parser("matrix")
    compare = commands.add_parser("sim-compare")
    compare.add_argument("--runs", nargs="+", required=True)
    compare.add_argument("--output")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from nav_eval.plugins import Registry
        from nav_eval.storage import read_json, write_json
        if args.command == "plugins":
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
        elif args.command in {"plan", "run"}:
            from nav_eval.planning import resolve
            resources = read_json(args.resources) if args.resources else {}
            config = read_json(args.config)
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
            registry = Registry()
            result = []
            for benchmark in registry.list("benchmark"):
                for simulator in benchmark.get("bindings", {}):
                    for method in registry.list("method"):
                        config = {"method": method["id"], "benchmark": benchmark["id"],
                                  "simulator": simulator, "track": "standardized"}
                        try:
                            resolve(config, registry=registry)
                            result.append({**config, "status": "declared_compatible", "runtime_verified": False})
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
        return 1 if isinstance(result, dict) and result.get("status") == "incomplete" else 0
    except (ValueError, KeyError, OSError, TimeoutError, ImportError, AttributeError, RuntimeError) as error:
        print(f"nav-eval: {error}", file=sys.stderr)
        return 2
