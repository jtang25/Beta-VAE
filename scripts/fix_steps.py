import sys
import json

if len(sys.argv) != 3:
    print("usage: python fix_steps.py input.log output.log")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

current_step = 0

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

            if data.get("phase") == "train":
                current_step += 25
                data["step"] = current_step

            new_json = json.dumps(data, separators=(",", ": "))
            fout.write(prefix + new_json + "\n")
        else:
            fout.write(line)
