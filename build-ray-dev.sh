#!/bin/bash
docker run --rm -w /ray -v `pwd`:/ray -ti rayproject/arrow_linux_x86_64_base:latest /ray/python/build-wheel-fast.sh
./build.sh

