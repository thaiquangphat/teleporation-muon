load_wandb_config() {
  local config_path="${WANDB_CONFIG_PATH:-./wandb.yaml}"
  if [[ ! -f "$config_path" ]]; then
    if [[ -n "${WANDB_API_KEY:-}" && -n "${WANDB_PROJECT:-}" ]]; then
      export WANDB_MODE="${WANDB_MODE:-online}"
      return 0
    fi
    printf 'W&B config not found: %s (copy wandb.yaml.example to wandb.yaml)\n' "$config_path" >&2
    return 1
  fi

  local exported_vars
  if ! exported_vars="$(python3 - "$config_path" <<'PY'
import shlex
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file) or {}

if not isinstance(config, dict):
    raise SystemExit("W&B YAML config must contain a mapping.")

api_key = str(config.get("api_key") or "").strip()
project = str(config.get("project") or "").strip()
entity = str(config.get("entity") or "").strip()
mode = str(config.get("mode") or "online").strip()

if not project:
    raise SystemExit("Set 'project' in wandb.yaml.")
if mode not in {"online", "offline", "disabled"}:
    raise SystemExit("'mode' in wandb.yaml must be online, offline, or disabled.")
if mode == "online" and not api_key:
    raise SystemExit("Set 'api_key' in wandb.yaml before using W&B online mode.")

for key, value in (
    ("WANDB_API_KEY", api_key),
    ("WANDB_PROJECT", project),
    ("WANDB_ENTITY", entity),
    ("WANDB_MODE", mode),
):
    print(f"export {key}={shlex.quote(value)}")
PY
  )"; then
    return 1
  fi
  eval "$exported_vars"
}
