ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG ORT_VERSION=1.22.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder
ARG ORT_VERSION

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential cmake pkg-config clang libclang-dev protobuf-compiler libssl-dev libopus-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:${PATH}

WORKDIR /app
COPY Cargo.toml ./
COPY src ./src
COPY static ./static

# whisper-rs builds whisper.cpp with CUDA through the Cargo feature.
ENV WHISPER_DONT_GENERATE_BINDINGS=1
RUN cargo build --release

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime
ARG ORT_VERSION

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libgomp1 libssl3 libopus0 \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime CPU shared library for Silero VAD.
# Version 1.22.0 is used because the Linux x64 .tgz asset exists with this URL pattern.
RUN mkdir -p /opt/onnxruntime && \
    curl -L --fail -o /tmp/onnxruntime.tgz \
      https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz && \
    tar -xzf /tmp/onnxruntime.tgz -C /opt/onnxruntime --strip-components=1 && \
    rm /tmp/onnxruntime.tgz

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:${LD_LIBRARY_PATH}
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
ENV RUST_LOG=info

WORKDIR /app
COPY --from=builder /app/target/release/audio2mqtt /usr/local/bin/audio2mqtt
COPY static ./static
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 15000 8080
ENTRYPOINT ["/entrypoint.sh"]
