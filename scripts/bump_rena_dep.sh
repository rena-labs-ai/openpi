#!/usr/bin/env bash
# Replace third_party/<dep> with the rena repo's copy at <commit>, and pin it in deps/<dep>.pin.json. Rena training
# imports these (rena_policy.AutolabelPrompt samples its prompts with autolabel_prompts), so a checkpoint's commit fixes
# the prompt mixture it trained with.
#   scripts/bump_rena_dep.sh autolabel-prompts|rena-task-grammar <commit>
set -euo pipefail
dep=${1:?usage: scripts/bump_rena_dep.sh autolabel-prompts|rena-task-grammar <commit>}
ref=${2:?usage: scripts/bump_rena_dep.sh autolabel-prompts|rena-task-grammar <commit>}
case $dep in
    autolabel-prompts) repo=rena-labs-ai/rena-autolabel path=packages/autolabel-prompts ;;
    rena-task-grammar) repo=rena-labs-ai/rena-task-grammar path= ;;
    *) echo "unknown dependency $dep" >&2; exit 1 ;;
esac
cd "$(git rev-parse --show-toplevel)"
copy=third_party/$dep
if [[ -n $(git status --porcelain --untracked-files=all -- "$copy") ]]; then
    echo "$copy has local changes; commit or discard them before bumping" >&2
    exit 1
fi
git fetch -q "git@github.com:$repo.git" "$ref"
commit=$(git rev-parse "FETCH_HEAD^{commit}")
tree=$(git rev-parse "$commit:$path")
git rm -rqf --ignore-unmatch "$copy"
rm -rf "$copy"
git read-tree --prefix="$copy/" -u "$tree"
pin=deps/$dep.pin.json
printf '{"repo": "%s", "path": "%s", "commit": "%s", "tree": "%s"}\n' "$repo" "$path" "$commit" "$tree" > "$pin"
git add "$pin"
echo "$copy is $repo@${commit:0:7}${path:+:$path}"
