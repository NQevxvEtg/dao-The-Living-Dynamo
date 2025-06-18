#!/bin/bash

echo "🔥 Stopping all containers..."
docker stop $(docker ps -q) 2>/dev/null

echo "🧹 Removing all containers..."
docker rm -f $(docker ps -aq) 2>/dev/null

echo "🧼 Removing all images..."
docker rmi -f $(docker images -aq) 2>/dev/null

echo "🗑️ Removing all volumes..."
docker volume rm -f $(docker volume ls -q) 2>/dev/null

echo "🌐 Removing all non-default networks..."
docker network rm $(docker network ls | grep -vE 'bridge|host|none' | awk 'NR>1 {print $1}') 2>/dev/null

echo "♻️ Pruning build cache..."
docker builder prune -af --filter "until=0h"

echo "✅ Docker has been reset to a clean state."

docker system df

