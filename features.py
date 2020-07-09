import numpy as np

SPECIAL_CHARS = ' "\'`'
LABEL = 1
SNIPPET = 0
WORD_LIST_FILE = "common_words.csv"
WORD_LIST = ['self', 'bm', '0', 'prop', '1', 'verts', 'face', 'bmesh', 'faces', 'ops', 'normal', 'col', 'context', 'edges', '2', 'from', 'name', 'true', 'row', 'x', 'v', 'y', 'other', 'box',
             'description', 'geom', 'facemap', 'default', 'bpy', 'min', 'layout', 'max', 'cls', 'align', 'size', 'type', 'f', 'edge', 'floatproperty', 'else', 'p', 'co', 'return', 'list', 'props',
             'filter_geom', 'vec', 'types', 'column', 'offset', 'self', '0', 'args', '1', 'torch', 'x', 'none', 'int', 'parser', 'type', 'add_argument', 'help', 'str', 'else', '2', 'f', 'default',
             'model', 'true', 'tensor', 'nets', 'logging', 'np', 'fromespnet', 'b', 'false', '-1', 'pytorch_backend', 'nn', 'data', 'ilens', 'i', 'batch', 'dtype', 'device', 'info', 'len', 'odim',
             'float', '3', '5', 'ifself', 'loss', 'group', 'shape', 'idim', 'xs', 'espnet', 'module', '%', ';', 'self', '0', '1', 'hvd', 'tf', 'std', 'torch', 'size', 'tensor', '#', 'none', '2',
             'model', 'if', 'args', 'rank', 'name', 'optimizer', 'else', '\\', 'state', 'keras', 'entries', 'default', 'df', 'false', '3', 'true', 'fromhorovod', 'x', 'dtype', 'h', 'conststd', 'np',
             'print', 'parser', 'org', '#include', 'verbose', 'help', 'value', 'add_argument', 'the', 'www', 'apache', 'software', 'asis', 'common', 'run', 'self', 'none', '0', 'args', '**kwargs',
             'name', '1', '*args', 'false', 'from', 'f', 'np', 'a', 'str', 'c', 'logger', 'jina', 'else', 'class', 'data', 'type', 'true', 'index', 'ifself', 'b', 'request', 'returnself', 'int', 'v',
             'return', 'k', 'fromjina', 'descriptor', 'jina_pb2', 'path', 'add', 'os', '3', '%', '2', 'yaml', 'super', 'msg', 'cls', 'd', 'func', '_descriptor', 'ndarray', 'serialized_options',
             'containing_type', 'self', '0', 'this', 'e', '1', 't', 'name', 'fluid', 'i', 'function', 'r', 'layers', ';', '2', 'none', 'n', 'input', 'false', '3', 'true', 'a', 'data', 'np', 'else',
             'value', 'key', 'if', 'default', 'c', 's', 'v', 'args', 'x', 'o', 'path', 'shape', 'type', 'os', 'list', 'int', 'l', 'stride', 'image', '-1', '#', 'u', 'paramattr', 'param_attr',
             'initializer', 'append', 'self', 's', '0', 'l', '+', '1', '#', 'x', 'game', 'y', 'cards', 'rows', 'gi', 'talon', '2',
             '#************************************************************************', 'none', 'app', 'append', 'xs', '_', 'c', 'stack', 'r', 'frompysollib', 'ifself', '-1', '4', 'return',
             'foriinrange', 'gameinfo', 'kw', 'suit', 'ys', 'canvas', 'foundations', '3', 'i', 'else', 'text', 'registergame', 'w', '\\', 'rank', 'label', 'frames', 'dealrow', 'from_stack', 'layout',
             'reserves', 'self', '0', '1', 'x', 'data', 'torch', 'edge_index', 'obj', '2', 'size', '`', 'dim', '-1', 'tensor', 'num_nodes', 'pos', 'batch', '3', 'nn', 'default', 'fromtorch_geometric',
             'none', 'optional', 'f', 'dataset', 'importtorch', 'tolist', 'out', 'model', 'device', 'out_channels', 'math', 'r', 'in_channels', '5', '\\mathbf', 'edge_attr', 'args', 'true', '4', 'y',
             'path', 'item', 'int', 'i', 'osp', 'to', 'row', 'dtype', 'cat']


# Alternate word list
# WORD_LIST = ['self', 'bm', '0', 'prop', '1', 'verts', 'face', 'bmesh', 'faces', 'ops', 'normal', 'col', 'context', 'edges', '2', 'from', 'name', 'true', 'row', 'x', 'v', 'y', 'other', 'box', 'description', 'self', '0', 'args', '1', 'torch', 'x', 'none', 'int', 'parser', 'type', 'add_argument', 'help', 'str', 'else', '2', 'f', 'default', 'model', 'true', 'tensor', 'nets', 'logging', 'np', 'fromespnet', 'b', ';', 'self', '0', '1', 'hvd', 'tf', 'std', 'torch', 'size', 'tensor', '#', 'none', '2', 'model', 'if', 'args', 'rank', 'name', 'optimizer', 'else', '\\', 'state', 'keras', 'entries', 'default', 'self', 'none', '0', 'args', '**kwargs', 'name', '1', '*args', 'false', 'from', 'f', 'np', 'a', 'str', 'c', 'logger', 'jina', 'else', 'class', 'data', 'type', 'true', 'index', 'ifself', 'b', 'self', '0', 'this', 'e', '1', 't', 'name', 'fluid', 'i', 'function', 'r', 'layers', ';', '2', 'none', 'n', 'input', 'false', '3', 'true', 'a', 'data', 'np', 'else', 'value', 'self', 's', '0', 'l', '+', '1', '#', 'x', 'game', 'y', 'cards', 'rows', 'gi', 'talon', '2', '#************************************************************************', 'none', 'app', 'append', 'xs', '_', 'c', 'stack', 'r', 'frompysollib', 'self', '0', '1', 'x', 'data', 'torch', 'edge_index', 'obj', '2', 'size', '`', 'dim', '-1', 'tensor', 'num_nodes', 'pos', 'batch', '3', 'nn', 'default', 'fromtorch_geometric', 'none', 'optional', 'f', 'dataset']


def score_vocab(snippet):
    """
    Takes a snippet of code and returns the score (1-0 discrete scale) for the snippet and each word in the word list
    :param snippet: A string representing a snippet of code
    :return: The score of the snippet for each word. A numpy array
    """
    # top_words = pd.read_csv(WORD_LIST_FILE, sep='\t', skiprows=0)
    word_count = np.char.count(snippet, WORD_LIST)
    return np.where(word_count > 0, 1, 0)


# input = ["if add_all","load_library.load_op_library(lib_file)","assert data.val_neg_edge_index.size() == (2, 2)"]


def output_design_matrix(input):
    """
    Produces the design matrix where each row is a snippet of code and its score for each word in the word list
    """
    return np.array([score_vocab(i) for i in input])
