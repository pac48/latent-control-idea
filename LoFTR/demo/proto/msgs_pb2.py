# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: msgs.proto

from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='msgs.proto',
  package='MatlabPython',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\nmsgs.proto\x12\x0cMatlabPython\"W\n\x05Numpy\x12\x10\n\x08num_dims\x18\x01 \x01(\x05\x12\x0c\n\x04\x64ims\x18\x02 \x03(\x05\x12 \n\x04type\x18\x03 \x01(\x0e\x32\x12.MatlabPython.Type\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\x0c*)\n\x04Type\x12\x0b\n\x07\x46LOAT64\x10\x00\x12\t\n\x05INT32\x10\x01\x12\t\n\x05UINT8\x10\x02\x62\x06proto3'
)

_TYPE = _descriptor.EnumDescriptor(
  name='Type',
  full_name='MatlabPython.Type',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='FLOAT64', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INT32', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='UINT8', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=117,
  serialized_end=158,
)
_sym_db.RegisterEnumDescriptor(_TYPE)

Type = enum_type_wrapper.EnumTypeWrapper(_TYPE)
FLOAT64 = 0
INT32 = 1
UINT8 = 2



_NUMPY = _descriptor.Descriptor(
  name='Numpy',
  full_name='MatlabPython.Numpy',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_dims', full_name='MatlabPython.Numpy.num_dims', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dims', full_name='MatlabPython.Numpy.dims', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='type', full_name='MatlabPython.Numpy.type', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data', full_name='MatlabPython.Numpy.data', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=28,
  serialized_end=115,
)

_NUMPY.fields_by_name['type'].enum_type = _TYPE
DESCRIPTOR.message_types_by_name['Numpy'] = _NUMPY
DESCRIPTOR.enum_types_by_name['Type'] = _TYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Numpy = _reflection.GeneratedProtocolMessageType('Numpy', (_message.Message,), {
  'DESCRIPTOR' : _NUMPY,
  '__module__' : 'msgs_pb2'
  # @@protoc_insertion_point(class_scope:MatlabPython.Numpy)
  })
_sym_db.RegisterMessage(Numpy)


# @@protoc_insertion_point(module_scope)
