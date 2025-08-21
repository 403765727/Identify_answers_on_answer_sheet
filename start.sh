#!/bin/bash

# 设置 Gunicorn 的配置参数
WORKERS=4
HOST="0.0.0.0"
PORT=5000

# 使用 Gunicorn 启动 Flask 应用
gunicorn -w $WORKERS -b $HOST:$PORT app:app > logs/app.log 2>&1
