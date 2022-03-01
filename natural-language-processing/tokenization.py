from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=15, oov_token='-')

teks = [
    'Saya suka programming',
    'Programming sangat menyenangkan!',
    'Machine Learning berbeda dengan pemrograman konvensional'
]

tokenizer.fit_on_texts(teks)
sequences = tokenizer.texts_to_sequences(teks)
# print(tokenizer.word_index)
# print(tokenizer.texts_to_sequences(['Saya suka programming']))
# print(tokenizer.texts_to_sequences(['Saya suka belajar programming sejak SMP']))

# sequences_samapanjang = pad_sequences(sequences)
sequences_samapanjang = pad_sequences(sequences, padding='post', maxlen=5, truncating='post')
print(sequences_samapanjang)