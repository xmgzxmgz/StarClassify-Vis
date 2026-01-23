#!/usr/bin/env bash
set -euo pipefail

docker compose down --remove-orphans
docker compose up -d --build

echo "已启动："
echo "前端 http://localhost:8080"
echo "后端 http://localhost:8000/docs"
