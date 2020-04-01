# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: CovidWorker.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='CovidWorker.proto',
  package='covidworker',
  syntax='proto3',
  serialized_options=_b('\n\035edu.uw.bhi.bionlp.covidworker'),
  serialized_pb=_b('\n\x11\x43ovidWorker.proto\x12\x0b\x63ovidworker\"$\n\x07Payload\x12\n\n\x02id\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\t\"#\n\x06Result\x12\n\n\x02id\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\t2\x8a\x01\n\x0b\x43ovidWorker\x12\x39\n\nRunMetaMap\x12\x14.covidworker.Payload\x1a\x13.covidworker.Result\"\x00\x12@\n\x11RunCovidPredictor\x12\x14.covidworker.Payload\x1a\x13.covidworker.Result\"\x00\x42\x1f\n\x1d\x65\x64u.uw.bhi.bionlp.covidworkerb\x06proto3')
)




_PAYLOAD = _descriptor.Descriptor(
  name='Payload',
  full_name='covidworker.Payload',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='covidworker.Payload.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='covidworker.Payload.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=34,
  serialized_end=70,
)


_RESULT = _descriptor.Descriptor(
  name='Result',
  full_name='covidworker.Result',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='covidworker.Result.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='covidworker.Result.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=72,
  serialized_end=107,
)

DESCRIPTOR.message_types_by_name['Payload'] = _PAYLOAD
DESCRIPTOR.message_types_by_name['Result'] = _RESULT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Payload = _reflection.GeneratedProtocolMessageType('Payload', (_message.Message,), {
  'DESCRIPTOR' : _PAYLOAD,
  '__module__' : 'CovidWorker_pb2'
  # @@protoc_insertion_point(class_scope:covidworker.Payload)
  })
_sym_db.RegisterMessage(Payload)

Result = _reflection.GeneratedProtocolMessageType('Result', (_message.Message,), {
  'DESCRIPTOR' : _RESULT,
  '__module__' : 'CovidWorker_pb2'
  # @@protoc_insertion_point(class_scope:covidworker.Result)
  })
_sym_db.RegisterMessage(Result)


DESCRIPTOR._options = None

_COVIDWORKER = _descriptor.ServiceDescriptor(
  name='CovidWorker',
  full_name='covidworker.CovidWorker',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=110,
  serialized_end=248,
  methods=[
  _descriptor.MethodDescriptor(
    name='RunMetaMap',
    full_name='covidworker.CovidWorker.RunMetaMap',
    index=0,
    containing_service=None,
    input_type=_PAYLOAD,
    output_type=_RESULT,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='RunCovidPredictor',
    full_name='covidworker.CovidWorker.RunCovidPredictor',
    index=1,
    containing_service=None,
    input_type=_PAYLOAD,
    output_type=_RESULT,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_COVIDWORKER)

DESCRIPTOR.services_by_name['CovidWorker'] = _COVIDWORKER

# @@protoc_insertion_point(module_scope)