#!/usr/bin/env bash

# Exit on first error
set -euo pipefail # 修正為 -euo pipefail，更嚴謹的錯誤處理

# Install Python dependencies
pip install -r requirements.txt

# Run Django migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --noinput