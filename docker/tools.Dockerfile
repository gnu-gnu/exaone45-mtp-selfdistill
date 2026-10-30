# CPU tools for data preparation, downloads and the generation client.
# Versions match the environment that produced data/SHA256SUMS.prompts.
FROM python:3.12-slim-bookworm
RUN python -m pip install --no-cache-dir \
      pyarrow==25.0.1 tokenizers==0.22.2 huggingface_hub==1.33.0 aiohttp==3.14.3
ENTRYPOINT []
