#!/usr/bin/env bash
# Commit what is actually running, then tell you how to push it.
#
# The repo already exists with real history; it just has no remote, and four
# weeks of changes have never been committed. This stages them, checks that
# nothing private is going along for the ride, and commits once.
#
# The check is the point. A tar of this directory is how an API key got
# shared once already, and a git push is the same mistake with a longer
# half-life — you can delete a tarball, but a commit is forever and public
# repositories get scraped within minutes.
#
#     bash scripts/repo_init.sh            # check, then commit
#     bash scripts/repo_init.sh --dry-run  # check only
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

GREEN=$'\033[1;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[1;31m'
BOLD=$'\033[1m'; OFF=$'\033[0m'
ok()   { printf '  %s✓%s %s\n' "$GREEN" "$OFF" "$1"; }
warn() { printf '  %s!%s %s\n' "$YELLOW" "$OFF" "$1"; }
die()  { printf '  %s✗%s %s\n' "$RED" "$OFF" "$1" >&2; exit 1; }

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

printf '\n%sCommitting what is running%s\n' "$BOLD" "$OFF"
git rev-parse --git-dir >/dev/null 2>&1 || die "this is not a git repository"
ok "repo at $PWD, branch $(git branch --show-current)"

# --- what would be committed -------------------------------------------
git add -A
STAGED=$(git diff --cached --name-only)
[[ -n "$STAGED" ]] || { warn "nothing to commit — already clean"; exit 0; }
COUNT=$(wc -l <<<"$STAGED")
ok "$COUNT file(s) staged"

# --- the safety check ---------------------------------------------------
# Belt: filenames that should never be in here.
BAD=$(grep -iE '(^|/)\.env($|\.)|(^|/)secrets/|\.db$|\.sqlite3?$|id_rsa|\.pem$|anthropic-key' <<<"$STAGED" || true)
if [[ -n "$BAD" ]]; then
    git reset -q
    printf '\n'
    die "these would have been committed and must not be:
$(sed 's/^/        /' <<<"$BAD")
      Nothing was committed. Fix .gitignore first."
fi
ok "no credential-shaped filenames"

# Braces: the content of what is staged. A key can hide in a file with a
# perfectly innocent name — that is exactly how the first one escaped.
LEAK=$(git diff --cached -U0 | grep -nE 'sk-ant-[A-Za-z0-9_-]{20,}' | head -3 || true)
if [[ -n "$LEAK" ]]; then
    git reset -q
    printf '\n'
    die "an API key appears in the staged changes. Nothing was committed.
      Find it, remove it, and rotate that key — assume it is burned."
fi
ok "no API key in the content"

SIZE=$(git diff --cached --numstat | awk '{a+=$1; d+=$2} END {print a"+ "d"-"}')
ok "diff: $SIZE lines"
printf '\n'
git diff --cached --stat | tail -25 | sed 's/^/    /'

if (( DRY )); then
    git reset -q
    printf '\n  %sdry run — nothing committed, nothing staged%s\n\n' "$YELLOW" "$OFF"
    exit 0
fi

# --- commit -------------------------------------------------------------
git commit -q -F - <<'MSG'
Four weeks of work: timers, diagnostics, status wall, printing

Everything built since "Make the rebuild reproducible", committed in one
go because none of it was committed as it went — which is the reason for
doing this at all.

  intervals   an exercise interval timer. Position is a pure function of
              elapsed time, so a late tick, a missed tick or a restart
              mid-round cannot make it wrong. Holds until you say go.
  selfcheck   twenty-six probes with a fix attached to each. Never hangs,
              never repairs anything, says nothing when all is well.
  monitor     the same probes on a rolling schedule, tiered by what they
              cost: the printer is USB and gets fifteen minutes, the
              paid API probes get six hours, a set lookup gets five
              seconds. Served at /status and resting on the appliance.
  printing    print_workout: a session as a sheet with boxes to tick, or
              the whole programme.
  casework    off. The machinery stays because workouts are built on it.
  screen      taking the screen is an event, not a property of every
              frame — which is why the board used to lose to a timer
              redraw within one second.

Tests: ten suites, including two that drive the real page in a browser,
because the last two bugs lived entirely in JavaScript and the Python
tests were perfectly happy throughout.
MSG

ok "committed: $(git log -1 --format='%h %s')"
cat <<TXT

  ${BOLD}Next, to get it onto GitHub:${OFF}

  1. Make an EMPTY PRIVATE repo at https://github.com/new
     Name it 'lance'. No README, no .gitignore, no licence — this repo
     already has all three and GitHub's would collide.

     PRIVATE matters: config.yaml has your city, your routines and your
     habits. No secrets, but it is yours.

  2. Point this repo at it and push:

     git remote add origin git@github.com:YOURNAME/lance.git
     git push -u origin master

     If that asks for a password, GitHub wants a key rather than one.
     Use HTTPS with a personal access token instead:

     git remote set-url origin https://github.com/YOURNAME/lance.git
     git push -u origin master

  3. Tell me it is up, and link your GitHub account to Claude so I can
     push changes to it. From then on your side of every update is:

     cd ~/assistant
     git fetch
     git diff HEAD origin/master     # read it before you take it
     git merge
     .venv/bin/python -m tests.run --quiet

TXT
