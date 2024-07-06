FROM debian:12

RUN uname -a
RUN apt-get update && apt-get install -y clang build-essential wget git sudo
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration
RUN git clone https://github.com/ethanc8/Gemini-Nano
RUN --mount=type=secret,id=hf_token \
    Gemini-Nano/ci/ci-script.sh bf16
