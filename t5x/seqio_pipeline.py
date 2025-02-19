import io
import os

import jsonlines
import seqio
import tensorflow as tf
import zstandard

#from  import GPT2Vocabulary
from t5x.data import gpt2_encoder

_GCS_BUCKET = 'gs://hugginghelen/t5x-test/pile'
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
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"paths": dl_paths['train']}),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={"paths": dl_paths['validation']}),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={"paths": dl_paths['test']}),
        ]

    def _generate_examples(self, paths):
        pipeline = PileReader(paths)
        for x, result in enumerate(pipeline):
            if result:
                idx = f'{x}_pile'
                yield idx, {'text': result['text']}


# Define a task
def register_dataset():
    _GCS_BUCKET = 'gs://hugginghelen/t5x-test'
    tfds.load(name="pile", data_dir=_GCS_BUCKET, split='train')
    seqio.TaskRegistry.add("pile",
                           seqio.TfdsDataSource(tfds_name="pile/lm:1.0.0"),
                           preprocessors=[
                               seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
                           ],
                           output_features={
                               'targets': seqio.Feature(gpt2_encoder.GPT2Vocabulary(), add_eos=True, dtype=tf.int32)
                           },
                           metric_fns=[]  # TODO(helen): do this.
                           )
    # task = seqio.TaskRegistry.get('pile')
    # ds = task.get_dataset(sequence_length=None, split="train", shuffle=False)
    # print('hello')

    # ds = seqio.get_dataset(mixture_or_task_name="pile",
    #                        task_feature_lengths={"inputs": 32, "targets": 32},
    #                        dataset_split="train",
    #                        shuffle=True,
    #                        feature_converter=seqio.DecoderFeatureConverter(pack=True)
    #                        )

# Define a FeatureConverter based on the model architecture.

# Use the top-level function seqio.get_dataset to obtain the tf.data.Dataset instance.

from transformers import GPT2TokenizerFast
import time

def simple_tokenization(item):
    return tokenizer.encode(tf.strings.as_string(item['text']).decode("utf-8"), return_tensors='tf')

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<|padding|>'})

st = time.time()
ds = tfds.load(name="pile", data_dir=_GCS_BUCKET, split='train')
print(f"Time to load dataset: {time.time() - st}")

st = time.time()
ds.map(lambda item: simple_tokenization(item), num_parallel_calls=10)


print('hello')