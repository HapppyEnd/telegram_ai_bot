import asyncio
import logging
import os

import torch
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
logging.basicConfig(level=logging.INFO)

API_TOKEN = os.getenv('BOT_TOKEN')
if not API_TOKEN:
    raise ValueError('BOT_TOKEN environment variable not found.')

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

model_name = 'facebook/opt-1.3b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    logging.warning('GPU is not available. Using CPU instead.')
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

if not model or not tokenizer:
    logging.error('Failed to load model or tokenizer.')

# Parameters for text generation
GENERATION_PARAMS = {
    'max_length': 200,
    'num_return_sequences': 1,
    'do_sample': True,
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 0.9,
    'repetition_penalty': 1.2,
    'pad_token_id': tokenizer.eos_token_id,
    'no_repeat_ngram_size': 2,
}


class UserInput(BaseModel):
    text: str

    @classmethod
    def validate(cls, input_text: str) -> bool:
        """Validates user input."""
        if not input_text:
            logging.warning('User input is empty.')
            return False
        if len(input_text) > 500:
            logging.warning('User input exceeds 500 characters.')
            return False
        return True


def get_main_keyboard() -> InlineKeyboardMarkup:
    """Creates the main keyboard for the bot."""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text='Generate Text',
                                  callback_data='generate_text'),]
        ]
    )
    return keyboard


@dp.message(Command('start'))
async def start_handler(msg: types.Message) -> None:
    """Handles the /start command."""
    logging.info('Received /start command')
    keyboard = get_main_keyboard()
    await msg.answer('Welcome! Press the button to generate text:',
                     reply_markup=keyboard)


@dp.callback_query(F.data == 'generate_text')
async def generate_text_callback(callback_query: types.CallbackQuery) -> None:
    """Handles the callback for generating text."""
    await callback_query.answer()
    await callback_query.message.answer(
        'Please enter the text for generation:')


@dp.message(F.text)
async def handle_user_input(msg: types.Message) -> None:
    """Handles user input for text generation."""
    input_text: str = msg.text.strip().replace('\n', ' ').replace('\r', ' ')
    logging.info(f'Received user input: {input_text}')

    try:
        # Использование Pydantic для валидации
        user_input = UserInput(text=input_text)
        if not UserInput.validate(input_text):
            await msg.answer('Invalid input. Please ensure your text is not '
                             'empty and does not exceed 500 characters.')
            return

        input_ids = tokenizer.encode(
            user_input.text,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            return_tensors='pt',
        ).to(device)

        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        logging.info('Starting text generation...')
        generated_text = await asyncio.to_thread(
            generate_text, input_ids, attention_mask)

        await msg.answer(f'Generated text: {generated_text}')

    except ValidationError as e:
        logging.error(f'Validation error: {e}', exc_info=True)
        await msg.answer('An unexpected error occurred. Please try again.')
    except Exception as e:
        logging.error(f'Error occurred during text generation: {e}',
                      exc_info=True)
        await msg.answer('An unexpected error occurred. Please try again.')


def generate_text(
        input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
    """Generates text using the model."""
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **GENERATION_PARAMS,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


async def main() -> None:
    """Starts the bot and begins polling for updates."""
    logging.info('Bot is running and listening...')
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
