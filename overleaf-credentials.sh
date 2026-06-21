#!/usr/bin/env bash
# Git credential helper for the Overleaf "report" submodule.
#
# Supplies the Overleaf Git token to git WITHOUT committing it: the token lives
# in .env (gitignored, chmod 600) next to this script, or in the environment.
#
# One-time setup (each teammate, on the cluster):
#   1) cp .env.example .env   # then put YOUR OWN token in .env, and: chmod 600 .env
#   2) git config credential."https://git.overleaf.com".helper \
#        /work/courses/3dv/team39/overleaf-credentials.sh
# Then:  git submodule update --init --remote report
# Pull/push the writeup from inside report/ as a normal git repo.
#
# Generate or revoke tokens at: Overleaf -> Account Settings -> Git Integration.

here="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${OVERLEAF_GIT_TOKEN:-}" ] && [ -f "$here/.env" ]; then
  # shellcheck disable=SC1091
  set -a; . "$here/.env"; set +a
fi

case "$1" in
  get)
    printf 'username=git\n'
    printf 'password=%s\n' "${OVERLEAF_GIT_TOKEN:-}"
    ;;
  *) : ;;  # store / erase: nothing to do
esac
