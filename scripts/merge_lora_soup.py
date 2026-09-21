"""LoRA weight averaging ("model soup") for adapters of any rank.

Pass --component <adapter_dir>:<weight> once per adapter. The result is a single adapter whose update is
dW_out = sum_i w_i * dW_i. PEFT's `cat` combination folds each adapter's alpha/r scaling into its A matrix,
so adapters of different rank merge exactly (the merged rank is the sum of ranks). The script verifies the
merged update against the weighted sum on one layer before saving.
"""
import argparse, json, os, shutil, torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--component", action="append", required=True, help="adapter_dir:weight")
parser.add_argument("--out", required=True)
args = parser.parse_args()
paths, weights = [], []
for item in args.component:
    path, weight = item.rsplit(":", 1)
    paths.append(path); weights.append(float(weight))
base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": "cpu"}, local_files_only=True)
model = PeftModel.from_pretrained(base, paths[0], adapter_name="c0", local_files_only=True)
for index, path in enumerate(paths[1:], start=1):
    model.load_adapter(path, adapter_name=f"c{index}", local_files_only=True)
names = [f"c{i}" for i in range(len(paths))]
model.add_weighted_adapter(names, weights, "soup", combination_type="cat")
model.set_adapter("soup")
layer = model.base_model.model.model.layers[10].self_attn.q_proj
def delta(name):
    return (layer.lora_B[name].weight.float() @ layer.lora_A[name].weight.float()) * layer.scaling[name]
expected = sum(w * delta(n) for w, n in zip(weights, names))
err = (delta("soup") - expected).abs().max().item()
assert err < 1e-3, f"merged update does not match the weighted sum: {err}"
model.save_pretrained(args.out, selected_adapters=["soup"], safe_serialization=True)
soup_dir = os.path.join(args.out, "soup")
for f in os.listdir(paths[0]):
    if f.startswith(("tokenizer", "vocab", "merges", "special_tokens", "added_tokens", "chat_template")):
        shutil.copy(os.path.join(paths[0], f), os.path.join(soup_dir, f))
json.dump({"base": args.base, "components": list(zip(paths, weights)), "combination": "cat", "max_abs_err_layer10_q_proj": err},
          open(os.path.join(soup_dir, "soup_recipe.json"), "w"), indent=2)
print("saved", soup_dir, "rank", layer.r["soup"], "err", err)
