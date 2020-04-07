# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from OpenNLP import OpenNLP_pb2 as OpenNLP_dot_OpenNLP__pb2


class OpenNLPStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.DetectSentences = channel.unary_unary(
        '/opennlp.OpenNLP/DetectSentences',
        request_serializer=OpenNLP_dot_OpenNLP__pb2.SentenceDetectionInput.SerializeToString,
        response_deserializer=OpenNLP_dot_OpenNLP__pb2.SentenceDetectionOutput.FromString,
        )


class OpenNLPServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def DetectSentences(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_OpenNLPServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'DetectSentences': grpc.unary_unary_rpc_method_handler(
          servicer.DetectSentences,
          request_deserializer=OpenNLP_dot_OpenNLP__pb2.SentenceDetectionInput.FromString,
          response_serializer=OpenNLP_dot_OpenNLP__pb2.SentenceDetectionOutput.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'opennlp.OpenNLP', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
