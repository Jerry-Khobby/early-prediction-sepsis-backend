import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes

# Create a logger instance
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)  # Capture debug logs

# Create a formatter and a console handler to output logs
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

app = FastAPI(
    title="Sepsis Prediction API",
    description="API for predicting sepsis risk and generating medical reports",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the main API router
app.include_router(routes.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
