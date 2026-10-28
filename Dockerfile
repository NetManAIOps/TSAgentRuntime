ARG BASE_IMAGE=docker.m.daocloud.io/library/python:3.12-slim
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ARG GYMTS_REPOSITORY=https://github.com/Jurmean/GymTS.git
ARG GYMTS_REVISION=5682bb9b0536788070369dfd3cbe1e6371319ed0
ARG SKTIME_REPOSITORY=https://github.com/NetManAIOps/sktime.git
ARG SKTIME_REVISION=080f4c3c3695f239e328548bac7d7a3ef7aa9127
ARG LLM_API_KEY=""
ARG AIPAAS_KEY=""

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    SKTIME_REPO=/opt/sktime \
    SKTIME_REVISION=${SKTIME_REVISION} \
    GYM_STORE=/data/gymts-state \
    HARNESS_JOURNAL=/data/gymts-harness/journal.jsonl \
    LLM_API_KEY=${LLM_API_KEY} \
    LLM_BASE_URL=https://kuaipao.pro/v1 \
    LLM_MODEL=deepseek-v4.1-flash-1017 \
    OTEL_SERVICE_NAME=ts-agent-runtime \
    OTEL_RESOURCE_ATTRIBUTES=acs.arms.service.feature=genai_app,gen_ai.instrumentation.sdk.name=loongsuite-genai-utils,service.version=1.5.0,deployment.environment=production \
    OTEL_TRACES_EXPORTER=otlp \
    OTEL_METRICS_EXPORTER=otlp \
    OTEL_LOGS_EXPORTER=none \
    OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://192.168.24.181:30012 \
    OTEL_EXPORTER_OTLP_HEADERS=aipaas-key=${AIPAAS_KEY} \
    OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental \
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY

WORKDIR /opt

RUN set -eux; \
    if python -c "import urllib.request;urllib.request.urlopen('http://mirrors.cloud.aliyuncs.com/debian/',timeout=3)" >/dev/null 2>&1; then \
        sed -i 's|deb.debian.org|mirrors.cloud.aliyuncs.com|g' /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null || true; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends git; \
    rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none "${GYMTS_REPOSITORY}" /opt/gymts-src \
    && git -C /opt/gymts-src checkout "${GYMTS_REVISION}" \
    && pip install /opt/gymts-src \
        /opt/gymts-src/adapters/timeseries \
        '/opt/gymts-src/adapters/libraryts[runtime,analysis]' \
    && rm -rf /opt/gymts-src

RUN git clone --filter=blob:none "${SKTIME_REPOSITORY}" /opt/sktime \
    && git -C /opt/sktime checkout "${SKTIME_REVISION}" \
    && pip install --no-deps -e /opt/sktime \
    && pip install -e /opt/sktime

RUN pip install \
        'loongsuite-otel-util-genai==0.9.1' \
        'opentelemetry-distro==0.66b1' \
        'opentelemetry-exporter-otlp-proto-http==1.45.1' \
        'googleapis-common-protos==1.70.0' \
        'protobuf>=5,<6'

COPY harness/harness.py /opt/harness/harness.py
COPY harness/index.html /opt/harness/index.html

VOLUME /data
EXPOSE 9000
CMD ["opentelemetry-instrument", "python", "/opt/harness/harness.py", "--serve"]
