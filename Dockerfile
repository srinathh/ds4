# DS4 Flash for gx10 (GB10 / sm_121 / CUDA 13.0 driver).
# make cuda-spark is the README-prescribed GB10 target. Multi-stage: the runtime
# nvidia/cuda flavor ships libcudart + libcublas; libcuda.so is injected at runtime
# by nvidia-container-toolkit.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    make gcc ca-certificates && rm -rf /var/lib/apt/lists/*
COPY . /ds4
WORKDIR /ds4
RUN make cuda-spark

FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04
COPY --from=build /ds4/ds4-server /usr/local/bin/ds4-server
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/ds4-server", "--cuda"]
