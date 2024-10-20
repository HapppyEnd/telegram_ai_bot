import asyncio
import base64
import logging
import os
from typing import Any, Dict

import torch
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import (FSInputFile, InlineKeyboardButton,
                           InlineKeyboardMarkup)
from dotenv import load_dotenv
from flymyai import FlyMyAIPredictException, client
from pydantic import BaseModel, constr
from transformers import pipeline

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Get API token
API_TOKEN = os.getenv('BOT_TOKEN')
if not API_TOKEN:
    raise ValueError('BOT_TOKEN environment variable not found.')
FLYMYAI_API_KEY = os.getenv('FLYMYAI_API_KEY')

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Determine device for text generation
device = 0 if torch.cuda.is_available() else -1
if device == -1:
    logging.warning('GPU is not available. Using CPU instead.')
else:
    logging.info('Using GPU for text generation.')

# Initialize text generator
model_name = 'facebook/opt-1.3b'
text_generator = pipeline('text-generation', model=model_name, device=device)

# Initialize image generator
image_model = "flymyai/SDTurboFMAAceleratedH100"

# Parameters for text generation
GENERATION_PARAMS: Dict[str, Any] = {
    'max_length': 200,
    'num_return_sequences': 1,
    'do_sample': True,
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 0.9,
    'repetition_penalty': 1.2,
    'truncation': True,
}

# Состояние пользователя
user_states = {}


class UserInput(BaseModel):
    """Model for user input validation."""
    text: constr(min_length=1, max_length=500)


def get_main_keyboard() -> InlineKeyboardMarkup:
    """Creates the main keyboard for the bot."""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(
                text='Generate Text', callback_data='generate_text')],
            [InlineKeyboardButton(
                text='Generate Image', callback_data='generate_image')]
        ]
    )
    return keyboard


@dp.message(Command('start'))
async def start_handler(msg: types.Message) -> None:
    """Handles the /start command."""
    logging.info('Received /start command')
    keyboard = get_main_keyboard()
    await msg.answer('Welcome! Choose an action:', reply_markup=keyboard)


@dp.callback_query(F.data == 'generate_text')
async def generate_text_callback(callback_query: types.CallbackQuery) -> None:
    """Handles the callback for generating text."""
    await callback_query.answer()
    await callback_query.message.answer(
        'Please enter the text for generation:')
    user_states[callback_query.from_user.id] = 'waiting_for_text'


def generate_text(input_text: str) -> str:
    """Generates text using the model."""
    output = text_generator(input_text, **GENERATION_PARAMS)
    return output[0]['generated_text']


@dp.callback_query(F.data == 'generate_image')
async def generate_image_callback(callback_query: types.CallbackQuery) -> None:
    """Handles the callback for generating images."""
    await callback_query.answer()
    await callback_query.message.answer(
        'Please enter a description for the image generation:')
    user_states[callback_query.from_user.id] = 'waiting_for_image'


@dp.message(F.text)
async def handle_user_input(msg: types.Message) -> None:
    """Handles user input for both text and image generation."""
    input_text: str = msg.text.strip().replace('\n', ' ').replace('\r', ' ')
    logging.info(f'Received user input: {input_text}')

    user_id = msg.from_user.id
    user_state = user_states.get(user_id)

    if user_state == 'waiting_for_text':
        try:
            logging.info('Starting text generation...')
            generated_text = await asyncio.to_thread(generate_text, input_text)
            await msg.answer(f'Generated text: {generated_text}')
        except Exception as e:
            logging.error(
                f'Error occurred during text generation: {e}', exc_info=True)
            await msg.answer('An unexpected error occurred. Please try again.')
        finally:
            del user_states[user_id]  # Сброс состояния
    elif user_state == 'waiting_for_image':
        logging.info('Starting image generation...')
        try:
            # Prepare the input data
            payload = {
                "prompt": input_text,
                "negative_prompt": "Dark colors, gloomy atmosphere, horror"
            }

            # Initialize the client
            fma_client = client(apikey=FLYMYAI_API_KEY)

            # Generate the image
            response = fma_client.predict(model=image_model, payload=payload)
            logging.info(f'Response from API: {response}')

            # Check if response contains the expected data
            if 'sample' in response.output_data and response.output_data[
                    'sample']:
                sample_encoded = response.output_data['sample'][0]
                sample = base64.b64decode(sample_encoded)

                # Save the image
                image_path = "generated_image.jpg"
                with open(image_path, "wb") as file:
                    file.write(sample)

                input_file = FSInputFile(image_path)
                await msg.answer_photo(input_file)
            else:
                logging.warning('No image was generated. Response did not '
                                'contain expected data.')
                await msg.answer('No image was generated. Please try again.')

        except FlyMyAIPredictException as e:
            logging.error(f"Error occurred during image generation: {e}",
                          exc_info=True)
            await msg.answer('An error occurred while generating the image.')

        except Exception as e:
            logging.error(f'Unexpected error during image generation: {e}',
                          exc_info=True)
            await msg.answer(
                'An unexpected error occurred during image generation.')
        finally:
            del user_states[user_id]  # Сброс состояния


async def main() -> None:
    """Starts the bot and begins polling for updates."""
    logging.info('Bot is running and listening...')
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
