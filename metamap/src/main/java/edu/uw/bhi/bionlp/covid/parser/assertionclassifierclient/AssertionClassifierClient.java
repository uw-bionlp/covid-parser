package edu.uw.bhi.bionlp.covid.parser.assertionclassifierclient;

import java.util.ArrayList;
import java.util.List;

import edu.uw.bhi.bionlp.covid.parser.assertionclassifier.AssertionClassifierGrpc;
import edu.uw.bhi.bionlp.covid.parser.assertionclassifier.AssertionClassifierGrpc.*;
import edu.uw.bhi.bionlp.covid.parser.assertionclassifier.AssertionClassifierOuterClass.*;
import io.grpc.Channel;
import io.grpc.ManagedChannelBuilder;

public class AssertionClassifierClient {

    private final AssertionClassifierBlockingStub blockingStub;

    public AssertionClassifierClient() {

        Channel channel = ManagedChannelBuilder.forAddress("localhost", 8080).usePlaintext(true).build();
        blockingStub = AssertionClassifierGrpc.newBlockingStub(channel);
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

    public String predictAssertion(String sentence, int beginIndex, int endIndex) {

        AssertionClassifierOutput output;
        AssertionClassifierInput request = AssertionClassifierInput.newBuilder()
            .setText(sentence)
            .setStartIndex(beginIndex)
            .setEndIndex(endIndex)
            .build();

        try {
            output = blockingStub.predictAssertion(request);
        } catch (Exception ex) {
            return "present";
        }
        return output.getPrediction();
    }
}