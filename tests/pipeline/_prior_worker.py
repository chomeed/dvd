"""Helper of tests/pipeline/test_prior_recipe.py: runs LeRobot's own configuration and construction code for
prior-training argvs in a separate process, on CPU. Not a test itself (the leading underscore keeps it out of
tests/run_all.sh).

  python tests/pipeline/_prior_worker.py JOBS_JSON        JOBS_JSON = [{"mode": ..., "out": <dir>, "argv": [...]}, ...]

  mode resolve  what `python -m lerobot.scripts.lerobot_train <argv>` resolves before it builds anything: its CLI parse
                (lerobot.configs.parser.wrap, which also handles --config_path and --resume), then
                TrainPipelineConfig.validate(), then <out>/train_config.json written by TrainPipelineConfig.save_pretrained,
                the call that writes train_config.json into every checkpoint
  mode build    the same, then what lerobot_train builds before its first step, from the dataset's meta/ only
                (prior/state9_dataset.py --meta-only is enough): load_dataset_metadata, the ImageNet image statistics that
                datasets.factory.make_dataset puts in place, make_policy, make_pre_post_processors. Writes what
                save_checkpoint writes except the weights (config.json, train_config.json now holding the dataset's
                features, both processors) and weights.json {tensor name: shape}
  mode wandb    the same parse, then LeRobot's own wandb logger (WandBLogger) started in wandb's disabled mode, first
                as LeRobot ships it and then after prior/lerobot_train.py's shim, plus one log_dict call; the outcomes
                go to <out>/wandb.json. wandb sends and writes nothing in disabled mode

The jobs run in order in this one process (the import of LeRobot and CLIP's loading code is paid once); the first
exception ends the process with its traceback. The caller hides CUDA (LeRobot then switches policy.device to cpu
with a warning), sets HF_HUB_OFFLINE=1 and points HF_DATASETS_CACHE at a temporary directory (the datasets library
writes an Arrow cache of every parquet file it reads). Third-party LeRobot plugins (register_third_party_plugins) are
not imported: none is involved without --env.type, and a clean environment of this repository has none. CLIP comes
from the Hugging Face cache (HF_HOME).
"""
import json
import sys
from pathlib import Path


def parse(argv: list):
    sys.argv = ["lerobot_train", *argv]                 # validate() re-reads --policy.path / --config_path from sys.argv
    import lerobot.scripts.lerobot_train  # noqa: F401  (every registration lerobot_train has done when it parses)
    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig

    @parser.wrap()
    def resolved(cfg: TrainPipelineConfig):
        return cfg

    cfg = resolved()
    cfg.validate()
    return cfg


def build(cfg, out: Path) -> None:
    import torch
    from lerobot.datasets.storage import load_dataset_metadata
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.utils.constants import IMAGENET_STATS

    meta = load_dataset_metadata(cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision,
                                 repo_type=cfg.dataset.repo_type)
    if cfg.dataset.use_imagenet_stats:                  # as datasets.factory.make_dataset does
        for key in meta.camera_keys:
            if key in meta.depth_keys:
                continue
            meta.stats.setdefault(key, {})
            for name, value in IMAGENET_STATS.items():
                meta.stats[key][name] = torch.tensor(value, dtype=torch.float32)
    policy = make_policy(cfg=cfg.policy, ds_meta=meta, rename_map=cfg.rename_map)
    pre, post = make_pre_post_processors(policy_cfg=cfg.policy, pretrained_path=None, pretrained_revision=None,
                                         dataset_stats=rename_stats(meta.stats, cfg.rename_map))
    policy.config.save_pretrained(out)
    cfg.save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)
    shapes = {name: list(tensor.shape) for name, tensor in policy.state_dict().items()}
    (out / "weights.json").write_text(json.dumps(shapes, indent=0, sort_keys=True))
    print(f"built {type(policy).__name__}: {len(shapes)} tensors, {sum(p.numel() for p in policy.parameters())} parameters")
    del policy, pre, post


def wandb_check(cfg) -> dict:
    import importlib.util
    import wandb
    from lerobot.common.wandb_utils import WandBLogger
    from wandb.sdk.wandb_run import Run

    path = Path(__file__).resolve().parents[2] / "prior" / "lerobot_train.py"
    spec = importlib.util.spec_from_file_location("prior_lerobot_train", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    cfg.wandb.mode = "disabled"
    report = {"wandb": wandb.__version__, "run_has_get_url": hasattr(Run, "get_url")}
    try:
        WandBLogger(cfg)
        report["without_shim"] = "ok"
    except AttributeError as error:
        report["without_shim"] = f"AttributeError: {error}"
    finally:
        wandb.finish()
    report["shim_added"] = launcher.add_get_url()
    logger = WandBLogger(cfg)
    logger.log_dict({"loss": 0.5}, step=1)
    wandb.finish()
    report["with_shim"] = "ok"
    return report


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: _prior_worker.py JOBS_JSON")
    for job in json.loads(Path(sys.argv[1]).read_text()):
        out = Path(job["out"])
        out.mkdir(parents=True, exist_ok=True)
        cfg = parse(job["argv"])
        if job["mode"] == "resolve":
            cfg.save_pretrained(out)
        elif job["mode"] == "build":
            build(cfg, out)
        elif job["mode"] == "wandb":
            (out / "wandb.json").write_text(json.dumps(wandb_check(cfg)))
        else:
            raise SystemExit(f"unknown mode {job['mode']!r}")
        print(f"{job['mode']}: wrote {out}", flush=True)


if __name__ == "__main__":
    main()
