package edu.uw.bhi.bionlp.predictor;

import edu.uw.bhi.bionlp.covidworker.CovidWorkerOuterClass.*;
import edu.uw.bhi.bionlp.covidworker.CovidWorkerGrpc.CovidWorkerImplBase;
import io.grpc.stub.StreamObserver;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

public class CovidWorkerImpl extends CovidWorkerImplBase {

    ObjectMapper jsonMapper = new ObjectMapper();
    NoteProcessor processor = new NoteProcessor();

    @Override
    public void runCovidPredictor(Payload request, StreamObserver<Result> responseObserver) {

        String input = request.getValue();
        String output = "";
        CovidResult processed = new CovidResult("");

        /**
         * Implement processing.
         */
        try {
            processed = processor.processNote(input);
        } catch (Exception ex) {
            System.out.println("Error: failed to process note. Exception: " + ex.getMessage());
        }

        /**
         * Convert output to JSON.
         */
        try {
            output = jsonMapper.writeValueAsString(processed);
        } catch (JsonProcessingException jex) {
            System.out.println("Error: failed to map object to JSON. Exception: " + jex.getMessage());
        }

        /**
         * Use gRPC builder to construct and return a Protobuffer object.
         */
        Result response = Result.newBuilder().setValue(output).setId(request.getId()).build();
        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }
}

// Call with: 
// mvn -DskipTests package exec:java -Dexec.mainClass=edu.uw.bhi.bionlp.predictor.App