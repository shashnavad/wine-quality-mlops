#!/usr/bin/env python3
"""
Script to run the FastAPI server for wine quality prediction.
"""

import os
import argparse
import uvicorn
from src.serving.model_server import app


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Wine Quality prediction server")
    
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to run the server on")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")
    parser.add_argument("--model-path", type=str, default="models/random_forest_model.joblib",
                        help="Path to the trained model")
    
    return parser.parse_args()


def main():
    """Main function to run the server."""
    args = parse_arguments()
    
    # Set model path environment variable
    os.environ["MODEL_PATH"] = args.model_path
    
    print(f"Starting server on {args.host}:{args.port}...")
    print(f"Using model from {args.model_path}")
    print("API documentation will be available at http://localhost:8080/docs")
    
    # Run the server
    uvicorn.run(
        "src.serving.model_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main() 