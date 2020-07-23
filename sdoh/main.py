from concurrent import futures

import grpc
import json

from grpc_server.CovidParser_pb2_grpc import SDoHServicer, add_SDoHServicer_to_server

from process import DocumentProcessor

class Servicer(SDoHServicer):

    def __init__(self):
        self.processor = DocumentProcessor()

    def Predict(self, request, context):

        # Process the note.
        prediction = self.processor.predict(request.text)
        prediction.id = request.id

        return prediction

server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
add_SDoHServicer_to_server(Servicer(), server)
server.add_insecure_port('[::]:8080')
server.start()
server.wait_for_termination()