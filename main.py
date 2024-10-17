import asyncio
import logging
import os
import re

import replicate
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

API_TOKEN = os.environ.get('BOT_TOKEN')
bot = Bot(token=API_TOKEN)
dp = Dispatcher()


def get_main_keyboard() -> InlineKeyboardMarkup:
    """
    Создает основную клавиатуру для бота.

    Returns:
        InlineKeyboardMarkup: Клавиатура с кнопкой для генерации текста.
    """
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Генерировать текст",
                              callback_data="generate_text")]
    ])
    return keyboard


@dp.message(Command("start"))
async def start_handler(msg: types.Message) -> None:
    """
    Обработчик команды /start.

    Args:
        msg (types.Message): Сообщение, содержащее команду /start.
    """
    logging.info("Получена команда /start")
    keyboard = get_main_keyboard()
    await msg.answer('Добро пожаловать! Выберите действие:',
                     reply_markup=keyboard)


@dp.callback_query(F.data == "generate_text")
async def generate_text_callback(callback_query: types.CallbackQuery) -> None:
    """
    Обработчик callback query для генерации текста.

    Args:
        callback_query (types.CallbackQuery): Callback query от пользователя.
    """
    await callback_query.message.answer(
        'Пожалуйста, введите текст для генерации:')
    await callback_query.answer()


@dp.message(F.text)
async def handle_user_input(msg: types.Message) -> None:
    """
    Обработчик пользовательского ввода текста.

    Args:
        msg (types.Message): Сообщение, содержащее текст от пользователя.
    """
    input_text: str = msg.text
    await msg.answer(f'Вы ввели: {input_text}. Генерирую текст...')

    try:
        # Использование replicate для генерации текста с моделью Meta-LLaMA
        model = replicate.models.get("meta/meta-llama-3-70b-instruct")
        output = replicate.run(model, input={"prompt": input_text})

        # Debugging: Print the output
        print("Output from Replicate API:", output)

        # Обработка вывода: фильтрация пустых строк и объединение фрагментов
        output = [element for element in output if element.strip()]
        generated_text: str = ' '.join(output)

        # Удаление лишних пробелов и знаков препинания
        generated_text = re.sub(r'\s+', ' ', generated_text)
        # Удалить завершающие знаки препинания
        generated_text = re.sub(r'[.!?]+', '', generated_text)

        if not generated_text:
            generated_text = 'Нет доступного текста для генерации.'

        await msg.answer(f'Сгенерированный текст: {generated_text}')
    except Exception as e:
        logging.error(f"Произошла ошибка при генерации текста: {e}",
                      exc_info=True)
        await msg.answer('Произошла ошибка при генерации текста.'
                         ' Пожалуйста, попробуйте еще раз.')


async def main() -> None:
    """
    Запускает бота.

    Эта функция инициализирует и запускает процесс опроса для бота.
    """
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
