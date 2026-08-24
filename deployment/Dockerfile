FROM python:3.10-slim
WORKDIR /opt/interface
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY app ./app
COPY configs ./configs
RUN python -m pip install --no-cache-dir .
ENV BE_WEB_HOST=0.0.0.0
ENV BE_WEB_PORT=8765
EXPOSE 8765
CMD ["python", "app/run_web_app.py"]
