import sys
import json

if len(sys.argv) != 3:
    print("usage: python fix_steps.py input.log output.log")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

last_train_step = None

with open(input_path, "r") as fin, open(output_path, "w") as fout:
    for line in fin:
        if "METRICS" in line and "{" in line:
            prefix, json_part = line.split("{", 1)
            json_str = "{" + json_part.strip()
            if json_str.endswith("\n"):
                json_str = json_str[:-1]
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                fout.write(line)
                continue

            phase = data.get("phase")

            if phase == "train":
                if "step" in data:
                    last_train_step = data["step"]

            elif phase == "val" and last_train_step is not None:
                data["step"] = last_train_step

            new_json = json.dumps(data, separators=(", ", ": "))
            fout.write(prefix + new_json + "\n")
        else:
            fout.write(line)
