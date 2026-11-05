#!/usr/bin/env python3
"""Universal experiment runner for FedRepRAG.

Usage:
    python run.py --config configs/slake.yaml --stage get_embedding
    python run.py --config configs/pubmedqa.yaml --stage fl_mlp_train

    # Inference stages require a checkpoint supplied by the user:
    python run.py --config configs/kilt_nq.yaml --stage mlp_infer \\
        --retrieval_mlp_ckpt output/KILT_NQ/mlp_train_centralized/mlp_stage2_epoch010.pt

    python run.py --config configs/pubmedqa.yaml --stage mlp_infer \\
        --mlp_ckpt output/PubMedQA/mlp_train_centralized/mlp_stage2_epoch020.pt

Any extra --KEY VALUE pairs override or extend the args defined in the YAML stage.
All paths are relative to the repository root (CODE_PTH).
"""
import argparse
import os
import subprocess
import sys

import yaml


def _is_path(value: str) -> bool:
    return any(value.startswith(p) for p in ("datasets/", "output/", "src/"))


def resolve(value, code_pth: str) -> str:
    s = str(value)
    return os.path.join(code_pth, s) if _is_path(s) else s


def build_cmd(script: str, args: dict, flags: list, code_pth: str) -> list:
    script_path = resolve(script, code_pth)
    if script.startswith("src/VQA/") and not os.path.isfile(script_path):
        script_path = os.path.join(code_pth, script[len("src/VQA/"):])
    cmd = [sys.executable, script_path]
    for key, val in (args or {}).items():
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{key}")
        elif key == "checkpoint" and val is not None:
            cmd += ["--checkpoint", os.path.join(code_pth, val)]
        elif val is not None:
            cmd += [f"--{key}", resolve(str(val), code_pth)]
    for flag in (flags or []):
        cmd.append(f"--{flag}")
    return cmd


def run_stage(stage_cfg: dict, code_pth: str, overrides: dict):
    script = stage_cfg["script"]
    args   = dict(stage_cfg.get("args", {}))
    flags  = stage_cfg.get("flags", [])
    src_dir = os.path.join(code_pth, "src")

    env = os.environ.copy()
    import_paths = [src_dir, code_pth]
    if env.get("PYTHONPATH"):
        import_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(import_paths)

    # Apply user overrides (e.g. checkpoint path passed on the command line)
    args.update(overrides)

    fc = stage_cfg.get("foreach_client")
    if fc:
        n = fc["n"]
        iter_args = fc.get("iter_args", {})
        for i in range(n):
            loop_args = dict(args)
            for key, val in iter_args.items():
                loop_args[key] = str(val).replace("{i}", str(i))
            cmd = build_cmd(script, loop_args, flags, code_pth)
            print(f"\n[Client {i}] " + " ".join(cmd))
            subprocess.run(cmd, cwd=src_dir, env=env, check=True)
    else:
        cmd = build_cmd(script, args, flags, code_pth)
        print("\n" + " ".join(cmd))
        subprocess.run(cmd, cwd=src_dir, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="FedRepRAG experiment runner",
        # Allow unknown args so users can pass --key value overrides
        add_help=True,
    )
    parser.add_argument("--config",   required=True, help="Path to a dataset YAML config")
    parser.add_argument("--stage",    required=True, help="Which stage to run")
    parser.add_argument("--code_pth", default=None,  help="Repo root (auto-detected if omitted)")
    opts, extra = parser.parse_known_args()

    # Parse extra args as key=value overrides: --key value
    overrides = {}
    it = iter(extra)
    for tok in it:
        if tok.startswith("--"):
            key = tok[2:]
            val = next(it, None)
            if val is None or val.startswith("--"):
                overrides[key] = True   # boolean flag
            else:
                overrides[key] = val
        else:
            print(f"Warning: unexpected token '{tok}' in extra args — ignored.")

    code_pth = (
        opts.code_pth
        or os.environ.get("CODE_PTH")
        or subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )

    config_path = opts.config if os.path.isabs(opts.config) else os.path.join(code_pth, opts.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    stages = cfg.get("stages", {})
    if opts.stage not in stages:
        print(f"Unknown stage '{opts.stage}'.\nAvailable: {list(stages.keys())}")
        sys.exit(1)

    run_stage(stages[opts.stage], code_pth, overrides)


if __name__ == "__main__":
    main()