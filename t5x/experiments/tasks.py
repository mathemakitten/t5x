import io
import os

import functools
import jsonlines
import seqio
import tensorflow as tf
import zstandard
from t5.evaluation import metrics

# For S3 only. This has the side effect of loading the plugin for the S3 filesystem.
#  See https://github.com/tensorflow/serving/issues/1963#issuecomment-1055903347.
import tensorflow_io

from t5x.data import gpt2_encoder

_GCS_BUCKET = 's3://hugginghelen/tfds' # 'gs://hugginghelen/t5x-test/pile'
os.environ['TFDS_DATA_DIR'] = _GCS_BUCKET
import tensorflow_datasets as tfds

#from the_pile import tfds_pile

try:
    import simdjson as json
except ImportError:
    print('Installing simdjson library')
    os.system('pip install -q pysimdjson')
    import simdjson as json

parser = json.Parser()

_DESCRIPTION = """
The Pile is a large, diverse, open source language modelling data set 
that consists of many smaller datasets combined together. 
The objective is to obtain text from as many modalities as possible to 
ensure that models trained using The Pile will have much broader generalization abilities.
We are currently developing Version 1, with an ultimate goal of 1 TiB of English text. 
After the completion of Version 1, our next goal is a fully-multilingual, 10TiB text dataset.
"""

_CITATION = """
"""
_DATASET_MODES = ["lm"]

_PILE_URL = 'https://the-eye.eu/public/AI/pile/train/{}.jsonl.zst'
_PILE_SPLITS = 8
lol = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
lmao = [str(j) for j in range(10, 30)]
# lol.pop(1)  # remove this for now it's crashing
_URLS = {
    'pile': {
        'train': [_PILE_URL.format(str(i).zfill(2)) for i in lol + lmao],
        'test': 'https://the-eye.eu/public/AI/pile/test.jsonl.zst',
        'validation': 'https://the-eye.eu/public/AI/pile/val.jsonl.zst',
    }
}

_VERSION = tfds.core.Version('1.0.0')
_RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
}

_NAME = 'pile'
_FILE_FORMAT = 'jsonlines'


def json_parser(x):
    try:
        line = parser.parse(x).as_dict()
        return line
    except ValueError:
        return x


class PileReader:
    def __init__(self, filenames, para_joiner='\n\n'):
        if not isinstance(filenames, list):
            filenames = [filenames]
        self.filenames = filenames
        self.para_joiner = para_joiner

    def _read_fn(self, filename):
        with tf.io.gfile.GFile(filename, 'rb+') as f:
            cctx = zstandard.ZstdDecompressor()
            reader_stream = io.BufferedReader(cctx.stream_reader(f))
            reader = jsonlines.Reader(reader_stream, loads=json_parser)
            for item in reader:
                result = dict()
                if isinstance(item['text'], str):
                    result['text'] = item['text']
                else:
                    text = item['text']
                    if isinstance(text, list):
                        text = self.para_joiner.join(text)
                        result['text'] = text
                yield result

    def __iter__(self):
        for filename in self.filenames:
            return self._read_fn(filename)


class ThePileConfig(tfds.core.BuilderConfig):
    def __init__(self, *, mode=None, **kwargs):
        super(ThePileConfig, self).__init__(
            name=mode,
            description="The Pile dataset",
            **kwargs)


class Pile(tfds.core.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ThePileConfig(version=_VERSION, mode=mode) for mode in _DATASET_MODES
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'text': tfds.features.Text()
            }),
            supervised_keys=("text", "text"),
            homepage='https://github.com/EleutherAI/The-Pile',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        dl_manager.verify_ssl = False
        dl_paths = dl_manager.download(_URLS['pile'])
        return {'train': self._generate_examples(path=dl_paths['train']),
                'validation': self._generate_examples(path=dl_paths['validation']),
                'test': self._generate_examples(path=dl_paths['test'])}

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _generate_examples(self, path):
        pipeline = PileReader(path)
        encoder = gpt2_encoder.GPT2Vocabulary()

        for x, result in enumerate(pipeline):
            if result:
                idx = f'{x}_pile'
                yield idx, {'text': result['text']} #{'text': self._int64_feature(encoder.encode(result['text']))}

import tensorflow_datasets as tfds

# Define the builder.
# builder = Pile()
# Make the builder store the data as a TFDS dataset.
# builder.download_and_prepare()
@seqio.map_over_dataset
def read_and_parse(x):
    # print(f"what is x??? {x}")
    return {'inputs': x['text'],
            'targets': x['text']
            }

# seqio.TaskRegistry.reset()

vocabulary = seqio.SentencePieceVocabulary(
    'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model', extra_ids=100)
output_features = {
    'inputs': seqio.Feature(vocabulary=vocabulary),
    'targets': seqio.Feature(vocabulary=vocabulary)
}

ds = tfds.load(name="pile", data_dir=_GCS_BUCKET, split='train')
seqio.TaskRegistry.add("pile",
                       seqio.TfdsDataSource(tfds_name="pile/lm:1.0.0"),
                       preprocessors=[ #functools.partial(translate, source_language='en', target_language='de')
                           functools.partial(read_and_parse),
                           seqio.preprocessors.tokenize,
                           # seqio.CacheDatasetPlaceholder(),
                           # seqio.preprocessors.append_eos
                       ],
                       output_features=output_features,
                        # TODO: inputs is unnecessary when using decoder featureconverter
                        #    'inputs': seqio.Feature(gpt2_encoder.GPT2Vocabulary(), add_eos=True, dtype=tf.int32),
                        #    'targets': seqio.Feature(gpt2_encoder.GPT2Vocabulary(), add_eos=True, dtype=tf.int32)
                       metric_fns=[metrics.bleu]  # TODO(helen): fix this.
                       )
task = seqio.TaskRegistry.get('pile')
ds = task.get_dataset(
    sequence_length={"inputs": 1024, "targets": 1024},
    # sequence_length=None,
    #                   split="train",
                      shuffle=False, use_cached=False)