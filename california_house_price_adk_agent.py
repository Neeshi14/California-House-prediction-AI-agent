from typing import Dict, Any, Optional
import os
import pickle
import numpy as np
from pydantic import BaseModel, ValidationError
import re
import asyncio

class HouseInput(BaseModel):
    rooms: int
    size: int
    income: int
    population: int

class CaliforniaHousePriceAgent:
    def __init__(self):
        self.model = self._load_model()
        # Keywords for natural language understanding
        self.param_patterns = {
            'rooms': [
                r'(\d+)[\s-]*(?:room|rooms|bedroom|bedrooms|bed|br)\b',
                r'(?:room|rooms|bedroom|bedrooms|bed)\s*(?:is|are|of)?\s*(\d+)'
            ],
            'size': [
                r'(\d+[,\d]*)\s*(?:sq\s*ft|square\s*feet|square\s*foot|sqft)',
                r'(?:size|area)\s*(?:is|of)?\s*(\d+[,\d]*)'
            ],
            'income': [
                r'\$?\s*(\d+[,\d]*k?)\s*(?:income|salary|earning)',
                r'(?:income|salary|earning)\s*(?:is|of)?\s*\$?\s*(\d+[,\d]*k?)'
            ],
            'population': [
                r'(\d+[,\d]*)\s*(?:population|people|residents)',
                r'(?:population|people|residents)\s*(?:is|of)?\s*(\d+[,\d]*)'
            ]
        }

    def _load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), 'california_house_price_model.pkl')
        print(f"Attempting to load model from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
            
        if os.path.getsize(model_path) == 0:
            print(f"Error: Model file is empty at {model_path}")
            return None
            
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                print("Model loaded successfully!")
                return model
        except PermissionError:
            print(f"Error: Permission denied when trying to access {model_path}")
            return None
        except pickle.UnpicklingError as e:
            print(f"Error: Failed to unpickle model file: {str(e)}")
            return None
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            return None

    def _extract_value(self, text: str, patterns: list) -> Optional[int]:
        text = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = match.group(1)
                # Remove commas from numbers
                value = value.replace(',', '')
                # Handle 'k' notation for thousands
                if 'k' in value.lower():
                    value = value.lower().replace('k', '000')
                return int(''.join(filter(str.isdigit, value)))
        return None

    def _extract_parameters(self, message: str) -> Dict[str, int]:
        params = {}
        
        # First check for structured format (key=value)
        pairs = re.findall(r'(\w+)\s*=\s*(\d+)', message.lower())
        if pairs:
            for key, value in pairs:
                if key in ['rooms', 'size', 'income', 'population']:
                    params[key] = int(value)
            return params

        # If no structured format found, extract from natural language
        for param, patterns in self.param_patterns.items():
            value = self._extract_value(message, patterns)
            if value:
                params[param] = value

        return params

    async def handle_message(self, context: Optional[Any], message: str) -> str:
        if not message or not message.strip():
            return self._get_help_message()
            
        # Handle greetings
        if any(greeting in message.lower() for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return "Hi! I'm your friendly AI assistant for California house price predictions. How can I help you today?"

        try:
            # Extract parameters from the message
            params = self._extract_parameters(message)
            
            # If no parameters found, provide help
            if not params:
                return """I understand you're asking about house prices. To help you, I need:
                
1. Number of rooms
2. House size (in square feet)
3. Area's median income
4. Area's population

For example, you can ask:
"What's the price of a 3-bedroom house with 1500 sq ft in an area with $50,000 income and 23,456 people?"
"""

            # Check for missing parameters
            required_params = ['rooms', 'size', 'income', 'population']
            missing = [p for p in required_params if p not in params]
            if missing:
                missing_names = {
                    'rooms': 'number of rooms',
                    'size': 'house size in square feet',
                    'income': 'median income',
                    'population': 'area population'
                }
                return f"I need a few more details to give you an accurate prediction. Could you tell me the {', '.join(missing_names[p] for p in missing)}?"

            try:
                validated = HouseInput(
                    rooms=params['rooms'],
                    size=params['size'],
                    income=params['income'],
                    population=params['population']
                )
            except ValidationError as ve:
                return f"Some of the values don't look quite right: {str(ve)}"

            if not self.model:
                return "I apologize, but I'm having trouble accessing my prediction model. Please try again later."

            # Make prediction
            features = [[
                validated.rooms,
                validated.size,
                validated.income,
                validated.population
            ]]
            
            prediction = self.model.predict(features)[0]
            return f"""Based on the details you provided:
- {validated.rooms} rooms
- {validated.size:,} square feet
- ${validated.income:,} median income
- {validated.population:,} population

I estimate the house price to be ${int(prediction):,}. 

Would you like to try another prediction?"""

        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}"

    def _get_help_message(self) -> str:
        return """I can help you predict house prices in California!

Just tell me about the house and area naturally, like:
"How much would a 3-bedroom house with 1500 square feet cost in an area where the income is $50,000 and 23,456 people live?"

Or you can use the simple format:
rooms=3, size=1500, income=50000, population=23456

I'll understand either way!"""

async def main():
    agent = CaliforniaHousePriceAgent()
    print("AI assistant is ready to help! Type 'quit' to exit.")
    print(agent._get_help_message())

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("AI: Goodbye! Have a great day!")
            break

        response = await agent.handle_message(None, user_input)
        print("AI:", response)

if __name__ == "__main__":
    asyncio.run(main())