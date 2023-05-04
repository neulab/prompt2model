import os
import sys
import subprocess
import re

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} MODEL_NAME [ORG_NAME]")
        sys.exit(1)

    model_name = sys.argv[1]
    org_name = sys.argv[2] if len(sys.argv) >= 3 else None

    if org_name:
        full_path = f"{org_name}/{model_name}"
    else:
        full_path = model_name

    subprocess.run(["git", "clone", f"https://huggingface.co/{full_path}"],
                   env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    model_bin_file = f"{model_name}/pytorch_model.bin"

    if os.path.exists(model_bin_file):
        with open(model_bin_file, "r") as file:
            content = file.read()
            size = re.search(r'size (\d+)', content).group(1)
            print(f"{full_path}\t{size}")
    else:
        model_bin_file = f"{model_name}/pytorch_model.bin.index.json"
        with open(model_bin_file, "r") as file:
            import json
            content = json.loads(file.read())
            size = content["metadata"]["total_size"]
            print(f"{full_path}\t{size}")

    subprocess.run(["rm", "-rf", model_name])

if __name__ == "__main__":
    main()
