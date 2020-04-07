package edu.uw.bhi.bionlp.covid.parser.grpcclient;

import java.util.ArrayList;
import java.util.List;

import edu.uw.bhi.bionlp.covid.parser.opennlp.OpenNLPGrpc;
import edu.uw.bhi.bionlp.covid.parser.opennlp.OpenNLPGrpc.*;
import edu.uw.bhi.bionlp.covid.parser.opennlp.OpenNLPOuterClass.*;
import io.grpc.Channel;
import io.grpc.ManagedChannelBuilder;

public class OpenNLPClient {

    private final OpenNLPBlockingStub blockingStub;

    public OpenNLPClient() {

        Channel channel = ManagedChannelBuilder.forAddress("open-nlp", 8080).usePlaintext(true).build();
        blockingStub = OpenNLPGrpc.newBlockingStub(channel);
    }

    public List<Sentence> detectSentences(String text) {

        SentenceDetectionOutput output;
        SentenceDetectionInput request = SentenceDetectionInput.newBuilder().setText(text).build();

        try {
            output = blockingStub.detectSentences(request);
        } catch (Exception ex) {
            return new ArrayList<Sentence>();
        }
        return output.getSentencesList();
    }
}