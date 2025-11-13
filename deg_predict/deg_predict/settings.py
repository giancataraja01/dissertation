"""
Django settings for deg_predict project.
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from Render
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Quick-start development settings - unsuitable for production
SECRET_KEY = "django-insecure-n--)(k*3ke+j=)!5#pha#+(pu7!sa74)nmk=i%7w!2-ibb%vfk"

# In production, set DEBUG=False (Render can inject an ENV var later)
DEBUG = True

#This is temporary
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

MIDDLEWARE.insert(1, "whitenoise.middleware.WhiteNoiseMiddleware")
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
#End of temporary

# Allow all hosts for now (you can restrict this later)
ALLOWED_HOSTS = ['*']


# Application definition
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_extensions",
    "dashboard",
    "login",
    "predict",
    "upload_dataset",
    "reports",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "deg_predict.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "deg_predict.wsgi.application"


# =====================================================
# DATABASE CONFIG FOR RENDER POSTGRESQL
# =====================================================

import dj_database_url

DATABASES = {
    'default': dj_database_url.config(
        default=os.getenv("DATABASE_URL"),
        conn_max_age=600,
        ssl_require=True
    )
}


# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True


# =====================================================
# STATIC FILES
# =====================================================

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]

# This is needed when deploying static files on Render / production
STATIC_ROOT = BASE_DIR / "staticfiles"


# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "/login/"
