FROM tiangolo/uvicorn-gunicorn:python3.9

RUN mkdir /fastapi

COPY requirements.txt /fastapi

WORKDIR /fastapi

COPY . /fastapi

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]