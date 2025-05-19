FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY arca_wordcount_bot.py .
COPY start.sh .

# Create directory for cookies
RUN mkdir -p /app/cookies && \
    chmod +x start.sh

# Print debug info
RUN echo "=== Container build complete ===" && \
    echo "Current directory: $(pwd)" && \
    echo "Directory contents:" && \
    ls -la

# Run the bot in single-run mode by default
# This can be overridden by passing different arguments
ENTRYPOINT ["/bin/sh", "-c", "echo '=== Starting container ===' && ./start.sh \"$@\"", "--"]
CMD ["-1"] 