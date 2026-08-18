import json
import os


def check_json(*path_segments) -> bool:
    load_path = os.path.join(*path_segments)

    if not load_path.endswith(".json"):
        load_path += ".json"

    return os.path.exists(load_path)


def load_json(*path_segments) -> dict:
    load_path = os.path.join(*path_segments)

    if not load_path.endswith(".json"):
        load_path += ".json"

    with open(load_path, 'r') as f:
        content = json.load(f)

    return content


def save_json(content, *path_segments) -> None:
    save_path = os.path.join(*path_segments)

    if not save_path.endswith(".json"):
        save_path += ".json"

    with open(save_path, 'w') as f:
        json.dump(content, f)
