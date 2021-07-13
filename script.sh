#!/bin/bash
containertag='parallel_ddp'
docker build . --tag $containertag &&\
docker run --rm -v $(pwd):/ddp $containertag ;
