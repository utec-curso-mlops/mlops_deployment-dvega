import json


def get_username():
    with open("/opt/ml/metadata/resource-metadata.json", "r") as f:
        app_metadata = json.loads(f.read())
        space_name = app_metadata["SpaceName"]
    return space_name