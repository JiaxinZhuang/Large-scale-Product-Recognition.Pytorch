#ler.  DO NOT EDIT!
# source: yt_example.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import yt_feature_pb2 as yt__feature__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='yt_example.proto',
  package='youtu_tf',
  syntax='proto3',
  serialized_pb=_b('\n\x10yt_example.proto\x12\x08youtu_tf\x1a\x10yt_feature.proto\"/\n\x07\x45xample\x12$\n\x08\x66\x65\x61tures\x18\x01 \x01(\x0b\x32\x12.youtu_tf.Features\"e\n\x0fSequenceExample\x12#\n\x07\x63ontext\x18\x01 \x01(\x0b\x32\x12.youtu_tf.Features\x12-\n\rfeature_lists\x18\x02 \x01(\x0b\x32\x16.youtu_tf.FeatureListsB\x03\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[yt__feature__pb2.DESCRIPTOR,])




_EXAMPLE = _descriptor.Descriptor(
  name='Example',
  full_name='youtu_tf.Example',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='features', full_name='youtu_tf.Example.features', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=48,
  serialized_end=95,
)


_SEQUENCEEXAMPLE = _descriptor.Descriptor(
  name='SequenceExample',
  full_name='youtu_tf.SequenceExample',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='context', full_name='youtu_tf.SequenceExample.context', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='feature_lists', full_name='youtu_tf.SequenceExample.feature_lists', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=97,
  serialized_end=198,
)

_EXAMPLE.fields_by_name['features'].message_type = yt__feature__pb2._FEATURES
_SEQUENCEEXAMPLE.fields_by_name['context'].message_type = yt__feature__pb2._FEATURES
_SEQUENCEEXAMPLE.fields_by_name['feature_lists'].message_type = yt__feature__pb2._FEATURELISTS
DESCRIPTOR.message_types_by_name['Example'] = _EXAMPLE
DESCRIPTOR.message_types_by_name['SequenceExample'] = _SEQUENCEEXAMPLE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Example = _reflection.GeneratedProtocolMessageType('Example', (_message.Message,), dict(
  DESCRIPTOR = _EXAMPLE,
  __module__ = 'yt_example_pb2'
  # @@protoc_insertion_point(class_scope:youtu_tf.Example)
  ))
_sym_db.RegisterMessage(Example)

SequenceExample = _reflection.GeneratedProtocolMessageType('SequenceExample', (_message.Message,), dict(
  DESCRIPTOR = _SEQUENCEEXAMPLE,
  __module__ = 'yt_example_pb2'
  # @@protoc_insertion_point(class_scope:youtu_tf.SequenceExample)
  ))
_sym_db.RegisterMessage(SequenceExample)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\370\001\001'))
# @@protoc_insertion_point(module_scope)

