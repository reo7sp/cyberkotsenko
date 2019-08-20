import logging

from prompt_toolkit import PromptSession
from prompt_toolkit import print_formatted_text, HTML

from cyberkotsenko.algos.simple_zadumchik import SimpleZadumchikGenerator

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s (%(filename)s:%(lineno)d)')
    session = PromptSession()

    generator = SimpleZadumchikGenerator.make_default()

    print_formatted_text(HTML('<b><i>READY</i></b>'))

    while True:
        msg = session.prompt('> ')
        print_formatted_text(HTML(f'<b>{generator.generate(msg)}</b>'))
