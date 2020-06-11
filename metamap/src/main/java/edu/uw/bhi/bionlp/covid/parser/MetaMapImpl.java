package edu.uw.bhi.bionlp.covid.parser;

import java.util.List;

import org.javatuples.Pair;

import edu.uw.bhi.bionlp.covid.parser.grpcclient.AssertionClassifierChannelManager;
import edu.uw.bhi.bionlp.covid.parser.CovidParser.MetaMapInput;
import edu.uw.bhi.bionlp.covid.parser.CovidParser.MetaMapOutput;
import edu.uw.bhi.bionlp.covid.parser.CovidParser.MetaMapSentence;
import edu.uw.bhi.bionlp.covid.parser.MetaMapGrpc.MetaMapImplBase;
import io.grpc.stub.StreamObserver;

/**
 *
 * @author ndobb
 */
public class MetaMapImpl extends MetaMapImplBase {

    DocumentProcessor processor;

    public MetaMapImpl(AssertionClassifierChannelManager assertionClassifierChannelManager) {
        this.processor = new DocumentProcessor(assertionClassifierChannelManager);
    }

    @Override
    public void extractNamedEntities(MetaMapInput request, StreamObserver<MetaMapOutput> responseObserver) {

        /*
         * Process. Output is a tuple<sentences,errors>.
         */
        Pair<List<MetaMapSentence>, List<String>> output = processor.processDocument(
            request.getSentencesList(), 
            request.getSemanticTypesList()
        );
        
        MetaMapOutput response = MetaMapOutput.newBuilder()
            .addAllSentences(output.getValue0())
            .addAllErrors(output.getValue1())
            .setId(request.getId())
            .build();

        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }
}