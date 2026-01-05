import yaml

with open("SE3D_PYPC.yml") as f:
    config = yaml.safe_load(f)

with open("requirements.txt", "w") as out:
    out.write("\n".join(config["dependencies"]))