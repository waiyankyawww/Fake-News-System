FROM python:3.11-slim

# Set the working directory
WORKDIR /app/src

# Copy the requirements and install them
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]



# streamlit==1.27.0
# numpy==1.26.4
# pandas==2.1.1
# scipy==1.11.3
# scikit-learn==1.3.2
# torch==2.8.0
# torchvision==0.23.0
# torchaudio==2.8.0
# nltk
# joblib
# regex
# transformers
# gensim
# tqdm
# # If you need seaborn, 0.13.x supports Py3.11:
# # seaborn==0.13.2
