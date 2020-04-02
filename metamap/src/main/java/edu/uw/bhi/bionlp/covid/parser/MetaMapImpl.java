package edu.uw.bhi.bionlp.covid.parser;

import java.util.List;

import edu.uw.bhi.bionlp.covid.parser.metamap.MetaMapGrpc.MetaMapImplBase;
import edu.uw.bhi.bionlp.covid.parser.metamap.MetaMapOuterClass.MetaMapInput;
import edu.uw.bhi.bionlp.covid.parser.metamap.MetaMapOuterClass.MetaMapOutput;
import edu.uw.bhi.bionlp.covid.parser.metamap.MetaMapOuterClass.MetaMapSentence;
import io.grpc.stub.StreamObserver;

/**
 *
 * @author ndobb
 */
public class MetaMapImpl extends MetaMapImplBase {

    DocumentProcessor processor = new DocumentProcessor();

    @Override
    public void extractNamedEntities(MetaMapInput request, StreamObserver<MetaMapOutput> responseObserver) {

        List<MetaMapSentence> output = processor.processDocument(request.getText());
        MetaMapOutput response = MetaMapOutput.newBuilder()
            .addAllSentences(output)
            .setId(request.getId())
            .setText(request.getText())
            .build();

        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }
}