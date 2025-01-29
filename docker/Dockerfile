FROM supervisely/base-py-sdk:6.73.287

RUN apt-get update && apt-get install -y wget

RUN mkdir -p /models

RUN wget -O /models/ViT-B-32.pt "https://huggingface.co/jinaai/clip-models/resolve/main/ViT-B-32.pt"

COPY dev_requirements.txt /req/requirements.txt

RUN pip install --no-cache-dir -r /req/requirements.txt

ENV JINA_HIDE_SURVEY=1