# use official Python base image
FROM python:3.11

# set working directory
WORKDIR /app

# copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all app files
COPY . . 

# # expose streamlit default port
# EXPOSE 8501

# run
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]