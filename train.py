import argparse

from omegaconf import OmegaConf

from style_transfer import run


def main(args):
    cfg = OmegaConf.load(f"./configs/{args.config}.yaml")
    run(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base_config")
    args, _ = parser.parse_known_args()

    main(args)
