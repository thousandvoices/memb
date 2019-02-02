## Мотивация
Несмотря на то, что векторные представления слов, набравшие популярность ещё во время появления word2vec,
сохраняют её до сих пор, существующие методы их хранения страдают от множества инженерных проблем.
Библиотека призвана решить две из них: большой размер файлов и медленную загрузку с диска в память.

Для уменьшения размера используется квантование, уже давно и успешно
[применяемое для сжатия нейронных сетей](https://www.tensorflow.org/performance/quantization),
с последующим кодированием Хаффмана. Реализация во многом основана на статье
[Deep Compression: Compressing Deep Neural Networks With Pruning, Trained Quantization and Huffman Coding](
https://arxiv.org/pdf/1510.00149.pdf). Эксперименты показывают, что сжатие в 6-8 раз почти не ухудшает качество
получившейся модели.

А для быстрой загрузки файлов все операции чтения идут через вызов `mmap`. Это позволяет:
* Открывать файлы почти мгновенно.
* Читать с диска только те вектора, которые нужны — и не расходовать на остальные время и свободную память.
* Если памяти в системе достаточно, те данные, которые уже прочитаны, останутся в памяти даже после перезапуска процесса.
Если памяти не хватает — они будут вытеснены операционной системой без усилий со стороны пользователя.

## Эксперименты
Все графики построены для модели tayga_upos_skipgram_300_2_2019, но  для ruscorpora_upos_cbow_300_20_2019 результаты получились
очень похожими. Легко заметить, что:
* Сжатая модель, использующая 6 бит на вес, по качеству неотличима от полной и занимает в 6 раз меньше места.
* Корреляция между размеченной вручную схожестью слов и полученной из модели гораздо менее чувствительна к шуму,
добавляемому квантованием, чем поиск аналогий — сжатые до 2 бит модели решают её так же хорошо, как и полные.
* И даже при переходе к 1 биту на вес модели сохраняют значительную часть предсказательной силы: корреляция падает всего
на пять процентов, точность при поиске аналогий — в 2 раза.
<p float="left">
  <img src="https://github.com/thousandvoices/memb/raw/add_readme/docs/images/spearman.png" alt="spearman" width="400" />
  <img src="https://github.com/thousandvoices/memb/raw/add_readme/docs/images/analogy.png" alt="analogy" width="400" />
  <img src="https://github.com/thousandvoices/memb/raw/add_readme/docs/images/sizes.png" alt="size" width="400" />
</p>

Опыты с англоязычными моделями показывают, что нейросети, обученные с векторами, сжатыми до 4 бит, достигают
тех же результатов, что и полные.

## Быстрый старт
* Скачайте и установите библиотеку со [страницы релизов](https://github.com/thousandvoices/memb/releases)
* Получите файлы со сжатыми моделями:
  * Скачайте один из файлов с https://drive.google.com/drive/folders/1L-iaHBHEeozbwx9OmceEr2JmzA2HKGRF. Сейчас доступны
  glove 840B и fasttext crawl для английского языка, а также tayga_upos_skipgram_300_2_2019 и ruscorpora_upos_cbow_300_20_2019
  из проекта [RusVectores](https://rusvectores.org/ru/models/) для русского.
  * Или воспользуйтесь утилитой memb_converter, которая преобразует файлы из текстового формата word2vec.
  Рекомендуемые параметры — `--quantization trained --bits-per-weight 6`. Параметр `--max-words` — необязательный,
  и его лучше оставить пустым.
* Теперь можно создать экземпляр класса Reader:
```python
from memb import Reader

reader = Reader('glove.840B.300d.4bit.bin')
```
  * Узнать длину векторов, которые он хранит:
```python
print(reader.dim)
```
  * Получить вектор для отдельного слова — массив размера (reader.dim,):
```python
print(reader['the'])
```
  * Или сразу для списка. В этом случае результат имеет размер (n_words, reader.dim):
```python
print(reader[['a', 'the', 'of']])
```

## Дополнительные возможности
Класс [`ReadersUnion`](https://github.com/thousandvoices/memb/blob/master/python/memb/readers_union.py#L41)
позволяет прозрачно объединить несколько источников векторов в один. Чтобы выбрать способ объединения, в конструктор надо
передать параметр `mode='average'` или `mode='concatenate'`.
```python
from memb import Reader, ReadersUnion

filenames = ['glove.840B.300d.4bit.bin', 'fasttext-crawl-300d-2M.4bit.bin']
readers = [Reader(filename) for filename in filenames]

union = ReadersUnion(readers, mode='concatenate')
print(union['the'])
```

Метод [`tokenizer_embedding`](https://github.com/thousandvoices/memb/blob/master/python/memb/reader.py#L61)
создает из объекта класса [`keras.preprocessing.text.Tokenizer`](https://keras.io/preprocessing/text/) матрицу,
которую можно передать в качестве весов в слой `Embedding`.
```python
from memb import Reader
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding

texts = ['First sentence', 'Second sentence']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

reader = Reader('glove.840B.300d.4bit.bin')

embedding_matrix = reader.tokenizer_embedding(tokenizer)
embedding_layer = Embedding(
    *embedding_matrix.shape,
    weights=[embedding_matrix],
    trainable=False)
```

`Reader` можно экспортировать в объект [`KeyedVectors`](https://radimrehurek.com/gensim/models/keyedvectors.html),
вызвав метод [`to_keyed_vectors`](https://github.com/thousandvoices/memb/blob/master/python/memb/reader.py#L14).
Разумеется, для того, чтобы он работал, нужно установить gensim.

## Сборка из исходников
Для сборки понадобятся:
* CMake
* Компилятор с поддержкой c++14
* И дополнительные библиотеки
  * Boost
  * Flatbuffers

Посмотреть шаги для установки зависимостей можно в скриптах для
[linux](https://github.com/thousandvoices/memb/blob/master/tools/development_image/install_deps_linux.sh) и
[mac os](https://github.com/thousandvoices/memb/blob/master/tools/development_image/install_deps_darwin.sh).
