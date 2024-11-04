from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import os
from dotenv import load_dotenv
import logging
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to allow only specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define the request body model
class ChatRequest(BaseModel):
    message: str

# Define the response model
class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        #logger.debug(f"Received request: {request}")
        # Use GPT to generate a response
        print(request.message)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": request.message}],
            model="gpt-4o",
            max_tokens=150
        )
        result = response['choices'][0].message.content
        print(result)
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return ChatResponse(response="Sorry, an error occurred while processing your request.")

# Run this FastAPI app using a command like:
# uvicorn filename:app --reload