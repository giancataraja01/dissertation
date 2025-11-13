services:
  - type: web
    name: django-deg-predict
    env: python
    rootDir: deg_predict
    buildCommand: "./build.sh"
    startCommand: "gunicorn deg_predict.wsgi:application"
