# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: CovidParser.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='CovidParser.proto',
  package='covid.parser',
  syntax='proto3',
  serialized_options=b'\n\036edu.uw.bhi.bionlp.covid.parser',
  serialized_pb=b'\n\x11\x43ovidParser.proto\x12\x0c\x63ovid.parser\"V\n\x08Sentence\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x18\n\x10\x62\x65gin_char_index\x18\x02 \x01(\x05\x12\x16\n\x0e\x65nd_char_index\x18\x03 \x01(\x05\x12\x0c\n\x04text\x18\x04 \x01(\t\"2\n\x16SentenceDetectionInput\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04text\x18\x02 \x01(\t\"n\n\x17SentenceDetectionOutput\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04text\x18\x02 \x01(\t\x12)\n\tsentences\x18\x03 \x03(\x0b\x32\x16.covid.parser.Sentence\x12\x0e\n\x06\x65rrors\x18\x04 \x03(\t\"]\n\x0cMetaMapInput\x12\n\n\x02id\x18\x01 \x01(\t\x12)\n\tsentences\x18\x02 \x03(\x0b\x32\x16.covid.parser.Sentence\x12\x16\n\x0esemantic_types\x18\x03 \x03(\t\"\xec\x01\n\x0eMetaMapConcept\x12\x0b\n\x03\x63ui\x18\x01 \x01(\t\x12\x14\n\x0c\x63oncept_name\x18\x02 \x01(\t\x12\x15\n\rsource_phrase\x18\x03 \x01(\t\x12\x16\n\x0esemantic_types\x18\x04 \x03(\t\x12\x1d\n\x15\x62\x65gin_sent_char_index\x18\x05 \x01(\x05\x12\x1b\n\x13\x65nd_sent_char_index\x18\x06 \x01(\x05\x12\x1c\n\x14\x62\x65gin_doc_char_index\x18\x07 \x01(\x05\x12\x1a\n\x12\x65nd_doc_char_index\x18\x08 \x01(\x05\x12\x12\n\nprediction\x18\t \x01(\t\"\x8d\x01\n\x0fMetaMapSentence\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x18\n\x10\x62\x65gin_char_index\x18\x02 \x01(\x05\x12\x16\n\x0e\x65nd_char_index\x18\x03 \x01(\x05\x12\x0c\n\x04text\x18\x04 \x01(\t\x12.\n\x08\x63oncepts\x18\x05 \x03(\x0b\x32\x1c.covid.parser.MetaMapConcept\"]\n\rMetaMapOutput\x12\n\n\x02id\x18\x01 \x01(\t\x12\x30\n\tsentences\x18\x02 \x03(\x0b\x32\x1d.covid.parser.MetaMapSentence\x12\x0e\n\x06\x65rrors\x18\x03 \x03(\t\"Z\n\x18\x41ssertionClassifierInput\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x18\n\x10start_char_index\x18\x02 \x01(\x05\x12\x16\n\x0e\x65nd_char_index\x18\x03 \x01(\x05\">\n\x19\x41ssertionClassifierOutput\x12\x12\n\nprediction\x18\x01 \x01(\t\x12\r\n\x05\x65rror\x18\x02 \x01(\t2i\n\x07OpenNLP\x12^\n\x0f\x44\x65tectSentences\x12$.covid.parser.SentenceDetectionInput\x1a%.covid.parser.SentenceDetectionOutput2Z\n\x07MetaMap\x12O\n\x14\x45xtractNamedEntities\x12\x1a.covid.parser.MetaMapInput\x1a\x1b.covid.parser.MetaMapOutput2|\n\x13\x41ssertionClassifier\x12\x65\n\x10PredictAssertion\x12&.covid.parser.AssertionClassifierInput\x1a\'.covid.parser.AssertionClassifierOutput\"\x00\x42 \n\x1e\x65\x64u.uw.bhi.bionlp.covid.parserb\x06proto3'
)




_SENTENCE = _descriptor.Descriptor(
  name='Sentence',
  full_name='covid.parser.Sentence',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='covid.parser.Sentence.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='begin_char_index', full_name='covid.parser.Sentence.begin_char_index', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end_char_index', full_name='covid.parser.Sentence.end_char_index', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='text', full_name='covid.parser.Sentence.text', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=35,
  serialized_end=121,
)


_SENTENCEDETECTIONINPUT = _descriptor.Descriptor(
  name='SentenceDetectionInput',
  full_name='covid.parser.SentenceDetectionInput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='covid.parser.SentenceDetectionInput.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='text', full_name='covid.parser.SentenceDetectionInput.text', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=123,
  serialized_end=173,
)


_SENTENCEDETECTIONOUTPUT = _descriptor.Descriptor(
  name='SentenceDetectionOutput',
  full_name='covid.parser.SentenceDetectionOutput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='covid.parser.SentenceDetectionOutput.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='text', full_name='covid.parser.SentenceDetectionOutput.text', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sentences', full_name='covid.parser.SentenceDetectionOutput.sentences', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='errors', full_name='covid.parser.SentenceDetectionOutput.errors', index=3,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=175,
  serialized_end=285,
)


_METAMAPINPUT = _descriptor.Descriptor(
  name='MetaMapInput',
  full_name='covid.parser.MetaMapInput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='covid.parser.MetaMapInput.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sentences', full_name='covid.parser.MetaMapInput.sentences', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='semantic_types', full_name='covid.parser.MetaMapInput.semantic_types', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=287,
  serialized_end=380,
)


_METAMAPCONCEPT = _descriptor.Descriptor(
  name='MetaMapConcept',
  full_name='covid.parser.MetaMapConcept',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='cui', full_name='covid.parser.MetaMapConcept.cui', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='concept_name', full_name='covid.parser.MetaMapConcept.concept_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='source_phrase', full_name='covid.parser.MetaMapConcept.source_phrase', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='semantic_types', full_name='covid.parser.MetaMapConcept.semantic_types', index=3,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='begin_sent_char_index', full_name='covid.parser.MetaMapConcept.begin_sent_char_index', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end_sent_char_index', full_name='covid.parser.MetaMapConcept.end_sent_char_index', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='begin_doc_char_index', full_name='covid.parser.MetaMapConcept.begin_doc_char_index', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end_doc_char_index', full_name='covid.parser.MetaMapConcept.end_doc_char_index', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='prediction', full_name='covid.parser.MetaMapConcept.prediction', index=8,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=383,
  serialized_end=619,
)


_METAMAPSENTENCE = _descriptor.Descriptor(
  name='MetaMapSentence',
  full_name='covid.parser.MetaMapSentence',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='covid.parser.MetaMapSentence.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='begin_char_index', full_name='covid.parser.MetaMapSentence.begin_char_index', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end_char_index', full_name='covid.parser.MetaMapSentence.end_char_index', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='text', full_name='covid.parser.MetaMapSentence.text', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='concepts', full_name='covid.parser.MetaMapSentence.concepts', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=622,
  serialized_end=763,
)


_METAMAPOUTPUT = _descriptor.Descriptor(
  name='MetaMapOutput',
  full_name='covid.parser.MetaMapOutput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='covid.parser.MetaMapOutput.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sentences', full_name='covid.parser.MetaMapOutput.sentences', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='errors', full_name='covid.parser.MetaMapOutput.errors', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=765,
  serialized_end=858,
)


_ASSERTIONCLASSIFIERINPUT = _descriptor.Descriptor(
  name='AssertionClassifierInput',
  full_name='covid.parser.AssertionClassifierInput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='covid.parser.AssertionClassifierInput.text', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start_char_index', full_name='covid.parser.AssertionClassifierInput.start_char_index', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end_char_index', full_name='covid.parser.AssertionClassifierInput.end_char_index', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=860,
  serialized_end=950,
)


_ASSERTIONCLASSIFIEROUTPUT = _descriptor.Descriptor(
  name='AssertionClassifierOutput',
  full_name='covid.parser.AssertionClassifierOutput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='prediction', full_name='covid.parser.AssertionClassifierOutput.prediction', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='error', full_name='covid.parser.AssertionClassifierOutput.error', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=952,
  serialized_end=1014,
)

_SENTENCEDETECTIONOUTPUT.fields_by_name['sentences'].message_type = _SENTENCE
_METAMAPINPUT.fields_by_name['sentences'].message_type = _SENTENCE
_METAMAPSENTENCE.fields_by_name['concepts'].message_type = _METAMAPCONCEPT
_METAMAPOUTPUT.fields_by_name['sentences'].message_type = _METAMAPSENTENCE
DESCRIPTOR.message_types_by_name['Sentence'] = _SENTENCE
DESCRIPTOR.message_types_by_name['SentenceDetectionInput'] = _SENTENCEDETECTIONINPUT
DESCRIPTOR.message_types_by_name['SentenceDetectionOutput'] = _SENTENCEDETECTIONOUTPUT
DESCRIPTOR.message_types_by_name['MetaMapInput'] = _METAMAPINPUT
DESCRIPTOR.message_types_by_name['MetaMapConcept'] = _METAMAPCONCEPT
DESCRIPTOR.message_types_by_name['MetaMapSentence'] = _METAMAPSENTENCE
DESCRIPTOR.message_types_by_name['MetaMapOutput'] = _METAMAPOUTPUT
DESCRIPTOR.message_types_by_name['AssertionClassifierInput'] = _ASSERTIONCLASSIFIERINPUT
DESCRIPTOR.message_types_by_name['AssertionClassifierOutput'] = _ASSERTIONCLASSIFIEROUTPUT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Sentence = _reflection.GeneratedProtocolMessageType('Sentence', (_message.Message,), {
  'DESCRIPTOR' : _SENTENCE,
  '__module__' : 'CovidParser_pb2'
  # @@protoc_insertion_point(class_scope:covid.parser.Sentence)
  })
_sym_db.RegisterMessage(Sentence)

SentenceDetectionInput = _reflection.GeneratedProtocolMessageType('SentenceDetectionInput', (_message.Message,), {
  'DESCRIPTOR' : _SENTENCEDETECTIONINPUT,
  '__module__' : 'CovidParser_pb2'
  # @@protoc_insertion_point(class_scope:covid.parser.SentenceDetectionInput)
  })
_sym_db.RegisterMessage(SentenceDetectionInput)

SentenceDetectionOutput = _reflection.GeneratedProtocolMessageType('SentenceDetectionOutput', (_message.Message,), {
  'DESCRIPTOR' : _SENTENCEDETECTIONOUTPUT,
  '__module__' : 'CovidParser_pb2'
  # @@protoc_insertion_point(class_scope:covid.parser.SentenceDetectionOutput)
  })
_sym_db.RegisterMessage(SentenceDetectionOutput)

MetaMapInput = _reflection.GeneratedProtocolMessageType('MetaMapInput', (_message.Message,), {
  'DESCRIPTOR' : _METAMAPINPUT,
  '__module__' : 'CovidParser_pb2'
  # @@protoc_insertion_point(class_scope:covid.parser.MetaMapInput)
  })
_sym_db.RegisterMessage(MetaMapInput)

MetaMapConcept = _reflection.GeneratedProtocolMessageType('MetaMapConcept', (_message.Message,), {
  'DESCRIPTOR' : _METAMAPCONCEPT,
  '__module__' : 'CovidParser_pb2'
  # @@protoc_insertion_point(class_scope:covid.parser.MetaMapConcept)
  })
_sym_db.RegisterMessage(MetaMapConcept)

MetaMapSentence = _reflection.GeneratedProtocolMessageType('MetaMapSentence', (_message.Message,), {
  'DESCRIPTOR' : _METAMAPSENTENCE,
  '__module__' : 'CovidParser_pb2'
  # @@protoc_insertion_point(class_scope:covid.parser.MetaMapSentence)
  })
_sym_db.RegisterMessage(MetaMapSentence)

MetaMapOutput = _reflection.GeneratedProtocolMessageType('MetaMapOutput', (_message.Message,), {
  'DESCRIPTOR' : _METAMAPOUTPUT,
  '__module__' : 'CovidParser_pb2'
  # @@protoc_insertion_point(class_scope:covid.parser.MetaMapOutput)
  })
_sym_db.RegisterMessage(MetaMapOutput)

AssertionClassifierInput = _reflection.GeneratedProtocolMessageType('AssertionClassifierInput', (_message.Message,), {
  'DESCRIPTOR' : _ASSERTIONCLASSIFIERINPUT,
  '__module__' : 'CovidParser_pb2'
  # @@protoc_insertion_point(class_scope:covid.parser.AssertionClassifierInput)
  })
_sym_db.RegisterMessage(AssertionClassifierInput)

AssertionClassifierOutput = _reflection.GeneratedProtocolMessageType('AssertionClassifierOutput', (_message.Message,), {
  'DESCRIPTOR' : _ASSERTIONCLASSIFIEROUTPUT,
  '__module__' : 'CovidParser_pb2'
  # @@protoc_insertion_point(class_scope:covid.parser.AssertionClassifierOutput)
  })
_sym_db.RegisterMessage(AssertionClassifierOutput)


DESCRIPTOR._options = None

_OPENNLP = _descriptor.ServiceDescriptor(
  name='OpenNLP',
  full_name='covid.parser.OpenNLP',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=1016,
  serialized_end=1121,
  methods=[
  _descriptor.MethodDescriptor(
    name='DetectSentences',
    full_name='covid.parser.OpenNLP.DetectSentences',
    index=0,
    containing_service=None,
    input_type=_SENTENCEDETECTIONINPUT,
    output_type=_SENTENCEDETECTIONOUTPUT,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_OPENNLP)

DESCRIPTOR.services_by_name['OpenNLP'] = _OPENNLP


_METAMAP = _descriptor.ServiceDescriptor(
  name='MetaMap',
  full_name='covid.parser.MetaMap',
  file=DESCRIPTOR,
  index=1,
  serialized_options=None,
  serialized_start=1123,
  serialized_end=1213,
  methods=[
  _descriptor.MethodDescriptor(
    name='ExtractNamedEntities',
    full_name='covid.parser.MetaMap.ExtractNamedEntities',
    index=0,
    containing_service=None,
    input_type=_METAMAPINPUT,
    output_type=_METAMAPOUTPUT,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_METAMAP)

DESCRIPTOR.services_by_name['MetaMap'] = _METAMAP


_ASSERTIONCLASSIFIER = _descriptor.ServiceDescriptor(
  name='AssertionClassifier',
  full_name='covid.parser.AssertionClassifier',
  file=DESCRIPTOR,
  index=2,
  serialized_options=None,
  serialized_start=1215,
  serialized_end=1339,
  methods=[
  _descriptor.MethodDescriptor(
    name='PredictAssertion',
    full_name='covid.parser.AssertionClassifier.PredictAssertion',
    index=0,
    containing_service=None,
    input_type=_ASSERTIONCLASSIFIERINPUT,
    output_type=_ASSERTIONCLASSIFIEROUTPUT,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_ASSERTIONCLASSIFIER)

DESCRIPTOR.services_by_name['AssertionClassifier'] = _ASSERTIONCLASSIFIER

# @@protoc_insertion_point(module_scope)
