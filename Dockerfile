# Use the official Python image as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose a port (optional if you're running a web server, e.g., Flask)
# EXPOSE 5000

# Define the command to run the application
CMD ["python", "GMF.py"]
