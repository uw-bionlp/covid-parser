package edu.uw.bhi.bionlp.covid.parser;

import java.util.List;

import org.javatuples.Pair;

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

        /*
         * Process. Output is a tuple<sentences,errors>.
         */
        Pair<List<MetaMapSentence>, List<String>> output = processor.processDocument(request.getSentencesList());
        
        MetaMapOutput response = MetaMapOutput.newBuilder()
            .addAllSentences(output.getValue0())
            .addAllErrors(output.getValue1())
            .setId(request.getId())
            .build();

        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }
}