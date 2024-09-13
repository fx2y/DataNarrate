FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variables
ENV PYTHONPATH=/app
ENV OPENAI_API_KEY=""
ENV SQLALCHEMY_DATABASE_URI=""

# Run app.py when the container launches
CMD ["uvicorn", "insightforge.database:app", "--host", "0.0.0.0", "--port", "8000"]
