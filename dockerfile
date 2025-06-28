# Use official Python image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y build-essential

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose the Streamlit port
EXPOSE 8501

RUN rm -f /app/enhancements/data_access/*.duckdb

# Default command to run the Streamlit app
CMD ["streamlit", "run", "enhancements/examples/pattern_analysis_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]