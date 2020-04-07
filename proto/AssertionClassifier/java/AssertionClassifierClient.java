package edu.uw.bhi.bionlp.covid.parser.grpcclient;

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

        Channel channel = ManagedChannelBuilder.forAddress("assertion-classifier", 8080).usePlaintext(true).build();
        blockingStub = AssertionClassifierGrpc.newBlockingStub(channel);
    }

    public List<Sentence> detectSentences(String text) throws Exception {

        SentenceDetectionOutput output;
        SentenceDetectionInput request = SentenceDetectionInput.newBuilder().setText(text).build();

        output = blockingStub.detectSentences(request);
        return output.getSentencesList();
    }

    public String predictAssertion(String sentence, int beginCharIndex, int endCharIndex) {

        AssertionClassifierOutput output;
        AssertionClassifierInput request = AssertionClassifierInput.newBuilder()
            .setText(sentence)
            .setStartIndex(beginCharIndex)
            .setEndIndex(endCharIndex)
            .build();

        try {
            output = blockingStub.predictAssertion(request);
        } catch (Exception ex) {
            return "present";
        }
        return output.getPrediction();
    }
}