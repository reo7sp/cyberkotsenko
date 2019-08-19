# from generator import Generator
from generator import Generator
import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from os import environ
from random import randint, choice
from string import ascii_lowercase


def get_random_string(length: int = 10):
    letters = ascii_lowercase
    return ''.join(choice(letters) for _ in range(length))


def write_msg(user_id, message):
    vk.method('messages.send', {'peer_id': user_id, 'message': message, 'random_id': str(randint(0, 2147483647))})


try:
    token = environ['API_KEY']
except:
    exit(1)

# Авторизуемся как сообщество
vk = vk_api.VkApi(token=token)
# Работа с сообщениями
longpoll = VkLongPoll(vk)

generator = Generator('trained_data/cyber_weights')
# Commander
print('Бот запущен')
# Основной цикл
for event in longpoll.listen():
    print(event.type)

    # Если пришло новое сообщение
    if event.type == VkEventType.MESSAGE_NEW:
        if hasattr(event, 'chat_id'):
            chat_id = event.chat_id
        else:
            chat_id = None

        print(event.text)
        # Если оно имеет метку для меня( то есть бота)
        if event.text.find('@cyberkotsenko') != -1 or event.to_me:

            # Сообщение от пользователя
            request: str = event.text

            pos = request.find('@cyberkotsenko')
            if pos != -1:
                request = request[pos:]
                request.replace('@cyberkotsenko', '')

            request = request.lower()

            if chat_id is not None:
                id = 2000000000 + chat_id
            else:
                id = event.user_id

            # Каменная логика ответа
            write_msg(id, generator.generate(seed=request, size=randint(10, 80)))
