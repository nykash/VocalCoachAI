#!/bin/bash
# Run breath_support.py using the project venv
cd "$(dirname "$0")"
source venv/bin/activate
exec python breath_support.py "$@"
