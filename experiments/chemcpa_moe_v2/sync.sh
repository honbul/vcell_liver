#!/usr/bin/env bash
# Fast-forward the server worktree onto the branch, clearing only the generated files that the
# incoming commit already contains byte for byte.
#
# Jobs write manifests and report tables into the worktree; those files are then committed from the
# laptop and come back on the next pull, where git refuses to overwrite an untracked file. This
# removes such a file only when its content is identical to the version being pulled in, and stops
# with a message otherwise. It never touches a file that differs.
set -euo pipefail
BRANCH=${1:-research/chemcpa-conditional-moe-v2}
git fetch -q origin
REF="origin/$BRANCH"
conflict=0
while IFS= read -r f; do
  [ -n "$f" ] || continue
  if git cat-file -e "$REF:$f" 2>/dev/null; then
    a=$(sha256sum "$f" | cut -d' ' -f1)
    b=$(git show "$REF:$f" | sha256sum | cut -d' ' -f1)
    if [ "$a" = "$b" ]; then rm -f "$f"; echo "cleared identical $f"
    else echo "DIFFERS, not touched: $f"; conflict=1; fi
  fi
done < <(git ls-files --others --exclude-standard)
[ "$conflict" -eq 0 ] || { echo "untracked files differ from $REF; resolve by hand"; exit 1; }
git merge -q --ff-only "$REF"
git log --oneline -1
# A job script is copied by sbatch at submission, so the checkout must be on the intended
# commit *before* anything is submitted. Print it so the submit line can be checked against it.
echo "server checkout now at $(git rev-parse --short HEAD) == $(git rev-parse --short "$REF")"
