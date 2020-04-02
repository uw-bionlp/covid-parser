package edu.uw.bhi.bionlp.covid.parser;

import java.util.ArrayList;
import java.util.List;

import edu.uw.bhi.bionlp.covid.parser.metamap.MetaMapGrpc.MetaMapImplBase;
import edu.uw.bhi.bionlp.covid.parser.metamap.MetaMapOuterClass.MetaMapInput;
import edu.uw.bhi.bionlp.covid.parser.metamap.MetaMapOuterClass.MetaMapOutput;
import io.grpc.stub.StreamObserver;

/**
 *
 * @author ndobb
 */
public class MetaMapImpl extends MetaMapImplBase {

    NoteProcessor processor = new NoteProcessor();

    @Override
    public void extractNamedEntities(MetaMapInput request, StreamObserver<MetaMapOutput> responseObserver) {
      
        String text = request.getText();
        processor.processNote(text);

    }
}