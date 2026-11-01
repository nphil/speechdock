"""Bounded, structural inspection of supported speech model files.

This checks the file layout and model schema, not the numerical quality or
trustworthiness of the weights. It never loads tensor contents into memory.
"""

import math
import os
import struct

MAX_MODEL_BYTES = 8 * 1024**3
_MAX_ITEMS = 100_000
_MAX_STRING = 16 * 1024 * 1024
# ggml type: (elements per block, bytes per block)
_TYPES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
          6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
          10: (256, 84), 11: (256, 110), 12: (256, 144),
          13: (256, 176), 14: (256, 210), 15: (256, 292)}


class _Reader:
    def __init__(self, file):
        self.file = file
        self.size = os.fstat(file.fileno()).st_size
        self.read_limit = self.size
        if not 0 < self.size <= MAX_MODEL_BYTES:
            raise ValueError('Model must be nonempty and at most 8 GiB')

    def read(self, n):
        if n < 0 or n > self.read_limit - self.file.tell():
            raise ValueError('Truncated model')
        data = self.file.read(n)
        if len(data) != n:
            raise ValueError('Truncated model')
        return data

    def skip(self, n):
        if n < 0 or n > self.read_limit - self.file.tell():
            raise ValueError('Truncated model')
        self.file.seek(n, 1)

    def unpack(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize('<' + fmt)))

    def string(self, length_fmt='Q', limit=_MAX_STRING):
        n, = self.unpack(length_fmt)
        if n > limit:
            raise ValueError('Model string is too large')
        try:
            return self.read(n).decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError('Invalid model string') from exc


def _tensor_size(dims, typ):
    if not dims or len(dims) > 4 or any(not 0 < d <= MAX_MODEL_BYTES for d in dims):
        raise ValueError('Invalid tensor dimensions')
    if typ not in _TYPES:
        raise ValueError('Unsupported tensor type')
    block, size = _TYPES[typ]
    if dims[0] % block:
        raise ValueError('Invalid quantized tensor dimensions')
    n = math.prod(dims) // block * size
    if n > MAX_MODEL_BYTES:
        raise ValueError('Tensor exceeds model size limit')
    return n


def _whisper_schema(h):
    vocab, ac, a, ah, al, tc, t, th, tl, mel, _ = h
    if (vocab not in (51864, 51865, 51866) or mel not in (80, 128)
            or ac != 1500 or tc != 448 or not 1 <= al <= 64 or not 1 <= tl <= 64
            or not 64 <= a <= 2048 or not 64 <= t <= 2048
            or not 1 <= ah <= 64 or not 1 <= th <= 64 or a % ah or t % th):
        raise ValueError('Unsupported Whisper architecture')
    s = {'encoder.positional_embedding': (a, ac), 'encoder.conv1.weight': (3, mel, a),
         'encoder.conv1.bias': (1, a), 'encoder.conv2.weight': (3, a, a),
         'encoder.conv2.bias': (1, a), 'decoder.positional_embedding': (t, tc),
         'decoder.token_embedding.weight': (t, vocab)}
    for prefix, width in [('encoder.ln_post', a), ('decoder.ln', t)]:
        for part in ('weight', 'bias'):
            s[f'{prefix}.{part}'] = (width,)
    for side, width, layers in [('encoder', a, al), ('decoder', t, tl)]:
        for i in range(layers):
            p = f'{side}.blocks.{i}.'
            for norm in ('attn_ln', 'mlp_ln') + (('cross_attn_ln',) if side == 'decoder' else ()):
                for part in ('weight', 'bias'):
                    s[p + norm + '.' + part] = (width,)
            for attn in ('attn',) + (('cross_attn',) if side == 'decoder' else ()):
                for projection in ('query', 'key', 'value', 'out'):
                    input_width = a if attn == 'cross_attn' and projection in ('key', 'value') else width
                    s[p + f'{attn}.{projection}.weight'] = (input_width, width)
                    if projection != 'key':
                        s[p + f'{attn}.{projection}.bias'] = (width,)
            s[p + 'mlp.0.weight'] = (width, 4 * width)
            s[p + 'mlp.0.bias'] = (4 * width,)
            s[p + 'mlp.2.weight'] = (4 * width, width)
            s[p + 'mlp.2.bias'] = (width,)
    return s


def _whisper(r):
    h = r.unpack('11i')
    schema = _whisper_schema(h)
    rows, cols = r.unpack('2i')
    if rows != h[9] or cols != 201:
        raise ValueError('Invalid Whisper mel filters')
    r.skip(rows * cols * 4)
    count, = r.unpack('i')
    if not 1 <= count <= h[0]:
        raise ValueError('Invalid Whisper vocabulary')
    for _ in range(count):
        length, = r.unpack('i')
        if not 0 <= length <= _MAX_STRING:
            raise ValueError('Invalid Whisper token')
        r.skip(length)
    seen = set()
    while r.file.tell() < r.size:
        nd, nl, typ = r.unpack('3i')
        if not 1 <= nd <= 4 or not 1 <= nl <= 255:
            raise ValueError('Invalid Whisper tensor header')
        dims = r.unpack(f'{nd}i')
        try:
            name = r.read(nl).decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError('Invalid tensor name') from exc
        if name in seen or name not in schema or dims != schema[name]:
            raise ValueError('Invalid Whisper tensor schema')
        seen.add(name)
        r.skip(_tensor_size(dims, typ))
    if seen != schema.keys():
        raise ValueError('Incomplete Whisper model')
    multi = h[0] >= 51865
    return dict(runtime='whisper.cpp', format='ggml', architecture='whisper', multilingual=multi,
                translation_capable=multi and not (h[4] == 32 and h[8] == 4), language_detection_capable=multi,
                requires_language=False)


_SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
            6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def _value(r, typ):
    if typ in _SCALARS:
        return r.unpack(_SCALARS[typ])[0]
    if typ == 8:
        return r.string()
    if typ == 9:
        item_type, count = r.unpack('IQ')
        if count > _MAX_ITEMS or item_type == 9:
            raise ValueError('Invalid metadata array')
        if item_type in _SCALARS:
            size = struct.calcsize('<' + _SCALARS[item_type])
            r.skip(size * count)
            return (item_type, count)
        if item_type == 8:
            for _ in range(count):
                r.string()
            return (item_type, count)
        raise ValueError('Unsupported metadata array')
    raise ValueError('Unsupported metadata type')


def _gguf(r):
    version, tensors, entries = r.unpack('IQQ')
    if version not in (2, 3) or not 1 <= tensors <= _MAX_ITEMS or entries > _MAX_ITEMS:
        raise ValueError('Unsupported GGUF header')
    metadata = {}
    types = {}
    r.read_limit = min(r.size, r.file.tell() + 64 * 1024 * 1024)
    for _ in range(entries):
        key = r.string(limit=1024)
        typ, = r.unpack('I')
        if not key or key in metadata:
            raise ValueError('Duplicate or empty metadata key')
        metadata[key] = _value(r, typ)
        types[key] = typ
    r.read_limit = r.size
    arch = metadata.get('general.architecture')
    if arch not in ('parakeet', 'canary'):
        raise ValueError('Unsupported GGUF architecture')
    alignment = metadata.get('general.alignment', 32)
    if type(alignment) is not int or not 1 <= alignment <= 4096 or alignment & (alignment - 1):
        raise ValueError('Invalid GGUF alignment')
    found = {}
    ranges = []
    for _ in range(tensors):
        name = r.string(limit=1024)
        nd, = r.unpack('I')
        if not name or name in found or not 1 <= nd <= 4:
            raise ValueError('Invalid GGUF tensor header')
        dims = r.unpack(f'{nd}Q')
        typ, offset = r.unpack('IQ')
        size = _tensor_size(dims, typ)
        if offset % alignment:
            raise ValueError('Invalid GGUF tensor offset')
        found[name] = (dims, typ)
        ranges.append((offset, offset + size))
    start = (r.file.tell() + alignment - 1) // alignment * alignment
    end = 0
    for first, last in sorted(ranges):
        if first < end or last > r.size - start:
            raise ValueError('Overlapping or truncated GGUF tensor')
        end = last
    _validate_crisp(arch, metadata, types, found)
    return dict(runtime='crispasr', format='gguf', architecture=arch,
                multilingual=arch == 'canary' or (metadata.get('parakeet.language') != 'en' and metadata.get('general.name') != 'Phonon-2'), translation_capable=False,
                language_detection_capable=False, requires_language=arch == 'canary')


def _validate_crisp(arch, m, types, tensors):
    # These are the legacy converter's explicit metadata names. Do not accept
    # unrelated GGUFs merely because they claim a speech architecture.
    prefix = arch + '.'
    keys = (['d_model', 'n_layers', 'n_heads', 'ff_dim', 'pred_hidden', 'pred_layers', 'joint_hidden', 'vocab_size', 'n_tdt_durations']
            if arch == 'parakeet' else
            ['d_model', 'enc_n_layers', 'dec_n_layers', 'n_heads', 'ff_dim', 'vocab_size', 'max_dec_ctx'])
    keys += ['sample_rate', 'n_mels', 'n_fft', 'win_length', 'hop_length', 'head_dim',
             'subsampling_factor', 'subsampling_channels', 'conv_kernel']
    for key in keys:
        name = prefix + key
        if types.get(name) != 4 or not 1 <= m[name] <= 100_000:
            raise ValueError('Missing or invalid speech model metadata')
    if types.get('tokenizer.ggml.tokens') != 9 or m['tokenizer.ggml.tokens'] != (8, m[prefix + 'vocab_size']):
        raise ValueError('Invalid speech tokenizer')
    width, ff, heads = (m[prefix + k] for k in ('d_model', 'ff_dim', 'n_heads'))
    layers = m[prefix + ('n_layers' if arch == 'parakeet' else 'enc_n_layers')]
    channels = m[prefix + 'subsampling_channels']
    kernel = m[prefix + 'conv_kernel']
    if (width > 4096 or ff > 16384 or heads > 128 or width % heads or layers > 64
            or m[prefix + 'head_dim'] != width // heads or channels > 2048
            or kernel > 255 or m[prefix + 'sample_rate'] != 16000
            or m[prefix + 'n_mels'] != 128 or m[prefix + 'n_fft'] != 512
            or m[prefix + 'win_length'] != 400 or m[prefix + 'hop_length'] != 160
            or m[prefix + 'subsampling_factor'] != 8):
        raise ValueError('Unsupported speech dimensions')
    if arch == 'canary' and 'canary.frame_dur_cs' in m and (
            types['canary.frame_dur_cs'] != 4 or m['canary.frame_dur_cs'] != 8):
        raise ValueError('Unsupported Canary frame duration')
    def need(name, shape=None, f32=False):
        item = tensors.get(name)
        if not item or (shape is not None and item[0] not in shape) or (f32 and item[1] != 0):
            raise ValueError('Missing or invalid speech tensor: ' + name)
    for i in (0, 2, 3, 5, 6):
        shape = (1, 1, channels, channels) if i in (3, 6) else (3, 3, 1, channels)
        need(f'encoder.pre.conv.{i}.weight', [shape])
        need(f'encoder.pre.conv.{i}.bias', [(channels,)], True)
    need('encoder.pre.out.weight', [(channels * 16, width)])
    need('encoder.pre.out.bias', [(width,)], True)
    for i in range(layers):
        p = f'encoder.layers.{i}.'
        for norm in ('norm_ff1', 'norm_attn', 'norm_conv', 'norm_ff2', 'norm_out'):
            for part in ('weight', 'bias'):
                need(p + norm + '.' + part, [(width,)], True)
        for ffname in ('ff1', 'ff2'):
            need(p + ffname + '.linear1.weight', [(width, ff)])
            need(p + ffname + '.linear2.weight', [(ff, width)])
        for proj in ('q', 'k', 'v', 'out', 'pos'):
            need(p + f'attn.{proj}.weight', [(width, width)])
        for part in ('u', 'v'):
            need(p + 'attn.pos_bias_' + part, [(width // heads, heads)], True)
        need(p + 'conv.pw1.weight', [(width, 2 * width), (1, width, 2 * width)])
        need(p + 'conv.pw2.weight', [(width, width), (1, width, width)])
        need(p + 'conv.dw.weight', [(kernel, 1, width)])
        for name in ('conv.dw.bias', 'conv.bn.weight', 'conv.bn.bias',
                     'conv.bn.running_mean', 'conv.bn.running_var'):
            need(p + name, [(width,)], True)
        if arch == 'canary':
            for name, size in [('ff1.linear1', ff), ('ff1.linear2', width),
                               ('ff2.linear1', ff), ('ff2.linear2', width),
                               ('attn.q', width), ('attn.k', width), ('attn.v', width),
                               ('attn.out', width), ('conv.pw1', 2 * width), ('conv.pw2', width)]:
                need(p + name + '.bias', [(size,)], True)
    if arch == 'parakeet':
        if m['parakeet.pred_layers'] != 2 or 'decoder.kda.0.q_proj.weight' in tensors:
            raise ValueError('Unsupported Parakeet predictor')
        hidden = m['parakeet.pred_hidden']
        joint = m['parakeet.joint_hidden']
        vocab = m['parakeet.vocab_size']
        output = vocab + 1 + m['parakeet.n_tdt_durations']
        need('decoder.embed.weight', [(hidden, vocab + 1)])
        for i in range(2):
            for part in ('w_ih', 'w_hh'):
                need(f'decoder.lstm.{i}.{part}', [(hidden, 4 * hidden)])
            for part in ('b_ih', 'b_hh'):
                need(f'decoder.lstm.{i}.{part}', [(4 * hidden,)], True)
        for name, inp, out in [('enc', width, joint), ('pred', hidden, joint), ('out', joint, output)]:
            need(f'joint.{name}.weight', [(inp, out)])
            need(f'joint.{name}.bias', [(out,)], True)
    else:
        if m['canary.dec_n_layers'] > 64:
            raise ValueError('Unsupported Canary decoder')
        vocab = m['canary.vocab_size']
        need('preprocessor.fb', [(257, 128)], True)
        need('preprocessor.window', [(400,)], True)
        need('decoder.embed.weight', [(width, vocab)])
        need('decoder.pos_enc', [(width, m['canary.max_dec_ctx'])], True)
        need('decoder.head.weight', [(width, vocab)])
        need('decoder.head.bias', [(vocab,)], True)
        for name in ('embed_ln', 'final_norm'):
            for part in ('weight', 'bias'):
                need(f'decoder.{name}.{part}', [(width,)], True)
        for i in range(m['canary.dec_n_layers']):
            p = f'decoder.layers.{i}.'
            for name in ('norm_sa', 'norm_ca', 'norm_ff'):
                for part in ('weight', 'bias'):
                    need(p + name + '.' + part, [(width,)], True)
            for name in ('sa_q', 'sa_k', 'sa_v', 'sa_out', 'ca_q', 'ca_k', 'ca_v', 'ca_out'):
                need(p + name + '.weight', [(width, width)])
                need(p + name + '.bias', [(width,)], True)
            need(p + 'ff_in.weight', [(width, ff)])
            need(p + 'ff_in.bias', [(ff,)], True)
            need(p + 'ff_out.weight', [(ff, width)])
            need(p + 'ff_out.bias', [(width,)], True)


def inspect_model(path):
    """Return capabilities for a structurally valid supported file, or raise ValueError."""
    with open(path, 'rb') as file:
        r = _Reader(file)
        magic = r.read(4)
        if magic == b'lmgg':
            return _whisper(r)
        if magic == b'GGUF':
            return _gguf(r)
        raise ValueError('Unsupported model format')
