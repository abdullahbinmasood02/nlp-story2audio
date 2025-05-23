# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: story_audio.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11story_audio.proto\x12\x0bstory_audio\"6\n\x0cStoryRequest\x12\x12\n\nstory_text\x18\x01 \x01(\t\x12\x12\n\nvoice_type\x18\x02 \x01(\t\";\n\x0b\x45motionData\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x0f\n\x07\x65motion\x18\x02 \x01(\t\x12\r\n\x05score\x18\x03 \x01(\x02\"\x8f\x01\n\rAudioResponse\x12\x0e\n\x06status\x18\x01 \x01(\t\x12\x15\n\raudio_content\x18\x02 \x01(\x0c\x12\x0e\n\x06\x66ormat\x18\x03 \x01(\t\x12\x13\n\x0bsample_rate\x18\x04 \x01(\x05\x12\x32\n\x10\x65motion_analysis\x18\x05 \x03(\x0b\x32\x18.story_audio.EmotionData2]\n\x11StoryAudioService\x12H\n\rGenerateAudio\x12\x19.story_audio.StoryRequest\x1a\x1a.story_audio.AudioResponse\"\x00\x62\x06proto3')



_STORYREQUEST = DESCRIPTOR.message_types_by_name['StoryRequest']
_EMOTIONDATA = DESCRIPTOR.message_types_by_name['EmotionData']
_AUDIORESPONSE = DESCRIPTOR.message_types_by_name['AudioResponse']
StoryRequest = _reflection.GeneratedProtocolMessageType('StoryRequest', (_message.Message,), {
  'DESCRIPTOR' : _STORYREQUEST,
  '__module__' : 'story_audio_pb2'
  # @@protoc_insertion_point(class_scope:story_audio.StoryRequest)
  })
_sym_db.RegisterMessage(StoryRequest)

EmotionData = _reflection.GeneratedProtocolMessageType('EmotionData', (_message.Message,), {
  'DESCRIPTOR' : _EMOTIONDATA,
  '__module__' : 'story_audio_pb2'
  # @@protoc_insertion_point(class_scope:story_audio.EmotionData)
  })
_sym_db.RegisterMessage(EmotionData)

AudioResponse = _reflection.GeneratedProtocolMessageType('AudioResponse', (_message.Message,), {
  'DESCRIPTOR' : _AUDIORESPONSE,
  '__module__' : 'story_audio_pb2'
  # @@protoc_insertion_point(class_scope:story_audio.AudioResponse)
  })
_sym_db.RegisterMessage(AudioResponse)

_STORYAUDIOSERVICE = DESCRIPTOR.services_by_name['StoryAudioService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _STORYREQUEST._serialized_start=34
  _STORYREQUEST._serialized_end=88
  _EMOTIONDATA._serialized_start=90
  _EMOTIONDATA._serialized_end=149
  _AUDIORESPONSE._serialized_start=152
  _AUDIORESPONSE._serialized_end=295
  _STORYAUDIOSERVICE._serialized_start=297
  _STORYAUDIOSERVICE._serialized_end=390
# @@protoc_insertion_point(module_scope)
