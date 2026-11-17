FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# The app is embedded in an iframe on another site (noemie.pro). Browsers block the
# third-party XSRF cookie there, which makes file uploads fail with a 403. The app has
# no login or user account to protect, so XSRF protection is disabled by default.
# Override with -e STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true when not embedding.
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py streamlit_app.py ./
COPY .streamlit/ .streamlit/

RUN useradd --create-home --uid 1000 documind  # uid 1000 is what Hugging Face Spaces expects
USER documind

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
