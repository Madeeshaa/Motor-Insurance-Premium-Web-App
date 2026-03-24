# Motor Insurance Premium Web Application

This folder contains a clean, professional HTML/CSS/JS frontend and a FastAPI backend designed to interface with the `premium_app_script.py` actuarial model.

## Prerequisites
Ensure you have the required Python packages for the backend:
```bash
pip install fastapi uvicorn pydantic
```
*(Plus any requirements from the original script like `torch`, `pandas`, `statsmodels`, `scikit-learn`).*

## How to Run

### Method 1: Using the provided `bat` scripts (Windows)
1. Double click **`run_backend.bat`**. This will start the FastAPI server on `http://localhost:8000`. Keep this terminal window open.
2. Double click **`run_frontend.bat`**. This will start a local HTTP server for the frontend on `http://localhost:8080` and you can open it in your browser.

### Method 2: Manual Terminal Commands
**1. Start the Backend**
Open a terminal in the `webapp` folder and run:
```bash
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```
*(The backend needs to be able to import `premium_app_script.py` from the parent directory, which is handled in `backend.py`.)*

**2. Start the Frontend**
Open a separate terminal in the `webapp` folder and run:
```bash
python -m http.server 8080
```
Then navigate to `http://localhost:8080` in your web browser.

## Features
- **FastAPI Core**: Asynchronous handling of complex model inference, wrapping the PyTorch ANFIS models.
- **Glassmorphism UI**: High-end styling matching modern design trends, completely built with vanilla CSS.
- **Dynamic Risk Feedback**: Real-time risk indicator adapting based on pure premium vs portfolio average.
