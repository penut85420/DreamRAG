import fire

from dream.cli import cli_demo
from dream.data import process_data
from dream.web import web_demo

if __name__ == "__main__":
    fire_map = dict(
        process_data=process_data,
        cli_demo=cli_demo,
        web_demo=web_demo,
    )
    fire.Fire(fire_map)
