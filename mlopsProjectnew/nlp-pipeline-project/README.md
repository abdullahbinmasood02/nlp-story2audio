# NLP Pipeline with Text Classification

This project implements an end-to-end MLOps pipeline for text classification using the News Category Dataset.

## Project Overview

The project consists of four main components:

1. **Data Pipeline**: Using Airflow to orchestrate data processing
2. **Model Pipeline**: Using MLflow to track experiments and register models
3. **Serving API**: Using FastAPI to serve predictions
4. **Monitoring**: Using Prometheus and Grafana to monitor the model in production

## Getting Started

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Kaggle API access (for dataset download)

### Installation

1. Clone this repository
2. Install dependencies:


pip install -r requirements.txt

export KAGGLE_USERNAME=your_username export KAGGLE_KEY=your_key


Project Structure

nlp-pipeline-project/
├── data/                 # Data storage
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── airflow/              # Airflow DAGs and plugins
│   ├── dags/             # Airflow DAG definitions
│   └── plugins/          # Airflow plugins
├── src/                  # Core source code
│   ├── data/             # Data processing modules
│   ├── features/         # Feature engineering code
│   ├── models/           # Model training code
│   └── utils/            # Utility functions
├── notebooks/            # Jupyter notebooks for exploration
├── api/                  # FastAPI app
├── monitoring/           # Monitoring configuration
│   ├── prometheus/       # Prometheus config
│   └── grafana/          # Grafana dashboards
├── tests/                # Unit and integration tests
└── docker/               # Docker configuration



## Running the Pipeline

1. Start Airflow:


docker-compose -f docker/docker-compose.yml up -d airflow

airflow dags trigger data_ingestion_dag

uvicorn api.main:app --reload


4. Access the API documentation at http://localhost:8000/docs

## Monitoring

Access Grafana dashboards at http://localhost:3000

## Contributors

- Your Name

## License

MIT

