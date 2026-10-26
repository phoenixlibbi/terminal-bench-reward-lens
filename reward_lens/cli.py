import argparse

from reward_lens import __version__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="reward-lens")
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args(argv)
    parser.print_help()


if __name__ == "__main__":
    main()
