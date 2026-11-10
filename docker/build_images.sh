#!/bin/sh

cd "$(dirname $0)/../"
docker build . --tag openai/retro-build:win64-v2 --file docker/windows/Dockerfile
docker push openai/retro-build:win64-v2
