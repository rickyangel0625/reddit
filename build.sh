# build.sh
#!/usr/bin/env bash
# Exit on first error
set -e

# Install Python dependencies
pip install -r requirements.txt

# Run Django migrations
python manage.py migrate

# Collect static files (if you had any, though for this simple page可能不需要太多)
python manage.py collectstatic --noinput