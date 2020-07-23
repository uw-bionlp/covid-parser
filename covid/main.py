from concurrent import futures

import grpc
import json

from grpc_server.CovidParser_pb2_grpc import CovidServicer as BaseCovidServicer, add_CovidServicer_to_server

from process import DocumentProcessor

class CovidServicer(BaseCovidServicer):

    def __init__(self):
        self.processor = DocumentProcessor()

    def Predict(self, request, context):

        # Process the note.
        prediction = self.processor.predict(request.text)
        prediction.id = request.id

        return prediction

server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
add_CovidServicer_to_server(CovidServicer(), server)
server.add_insecure_port('[::]:8080')
server.start()
server.wait_for_termination()