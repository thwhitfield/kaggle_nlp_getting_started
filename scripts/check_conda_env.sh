#!/usr/bin/env bash
tmpfile=$(mktemp)
conda env export --no-builds > "$tmpfile"
if diff -q setup/environment.yml "$tmpfile" >/dev/null 2>&1; then
  rm -f "$tmpfile"
  exit 0
else
  echo "Conda environment is not up to date."
  rm -f "$tmpfile"
  exit 1
fi
