#!/usr/bin/env bash
# shellcheck disable=SC1091
#
# Reformats Python code belonging to the repository.
# The primary Linter with support for reformatting
# files is Black. Any reformatted code causes
# non-zero exit code (CI failure).
# Additionally, there can be secondary Linters
# used to raise warnings, e.g., Flake8

REPOROOT=$(realpath "$(dirname "$0")/../..")
PATTERN="./*.py"
LINTER=$(command -v black)

if [ "${LINTER}" ]; then
  FLAKE8=$(command -v flake8)
  FLAKE8_IGNORE="--ignore=E402,E501,F821"
  COUNT=0

  # If -i is passed, format all files according to type/pattern.
  if [ "-i" != "$1" ]; then
    # list files matching PATTERN and which are part of HEAD's changeset
    LISTFILES="git diff-tree --no-commit-id --name-only HEAD -r --"
  else
    LISTFILES="git ls-files"
  fi

  echo -n "Linting Python files... "
  cd "${REPOROOT}" || exit 1
  # list files matching PATTERN and which are part of HEAD's changeset
  for FILE in $(eval "${LISTFILES} ${PATTERN}"); do
    # Flake8: line-length limit of 79 characters (default)
    if ${LINTER} -q -l 79 "${FILE}"; then COUNT=$((COUNT+1)); fi
    if [ "${FLAKE8}" ]; then  # optional
      # no error raised for Flake8 issues
      WARNING=$(flake8 ${FLAKE8_IGNORE} "${FILE}")
      if [ "${WARNING}" ]; then
        if [ "${WARNINGS}" ]; then
          WARNINGS+=$'\n'"${WARNING}"
        else
          WARNINGS="${WARNING}"
        fi
      fi
    fi
  done

  # any modified file (Git) raises and error
  MODIFIED=$(eval "git ls-files -m ${PATTERN}")
  if [ "${MODIFIED}" ]; then
    echo "ERROR"
    echo
    echo "The following files are modified ($(${LINTER} --version)):"
    echo "${MODIFIED}"
    exit 1
  fi
  # optional warnings
  if [ "${WARNINGS}" ]; then
    echo "WARNING"
    echo
    echo "The following issues were found:"
    echo "${WARNINGS}"
    echo
  else
    echo "OK (${COUNT} files)"
  fi
else  # soft error (exit normally)
  echo "ERROR: missing Python-linter (${LINTER})."
fi
