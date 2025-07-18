"""
WSGI config for reddit project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os
from pathlib import Path  # <-- 確保這行有被引入
from django.core.wsgi import get_wsgi_application
from whitenoise import WhiteNoise

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reddit.settings')

application = get_wsgi_application()


PROJECT_ROOT = Path(__file__).resolve().parent.parent 

# 將 WhiteNoise 的 root 指向正確的 STATIC_ROOT 位置
application = WhiteNoise(application, root=os.path.join(PROJECT_ROOT, 'staticfiles'))