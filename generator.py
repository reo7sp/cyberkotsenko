from model import get_model
from ast import literal_eval
import numpy as np


class Generator:
    def __init__(self, file: str = 'cyber_weights'):
        with open('char_indices') as f:
            char_indices = f.read()
        self._char_indices = literal_eval(char_indices)

        with open('indices_char') as f:
            indices_char = f.read()
        self._indices_char = literal_eval(indices_char)

        self._chars = sorted(list(self._char_indices.keys()))

        self._maxlen = 40

        self._model = get_model(self._maxlen, len(self._chars))
        self._model.load_weights(file)

        # print(self._char_indices)
        # print(self._indices_char)
        # print(self._chars)

    def generate(self, size: int = 50, diversity: float = 0.2):
        generated = ''
        sentence = 'хорошего дня хорошего дня хорошего дня  '
        generated += sentence

        for i in range(size):
            x_pred = np.zeros((1, self._maxlen, len(self._chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, self._char_indices[char]] = 1.

            preds = self._model.predict(x_pred, verbose=0)[0]
            next_index = self._sample(preds, diversity)
            next_char = self._indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

        return generated[40:]

    @staticmethod
    def _sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


if __name__ == '__main__':
    generator = Generator()
    text = generator.generate()
    print(text)
