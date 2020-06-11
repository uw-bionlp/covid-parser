package edu.uw.bhi.bionlp.covid.parser.grpcclient;

import edu.uw.bhi.bionlp.covid.parser.AssertionClassifierGrpc;
import edu.uw.bhi.bionlp.covid.parser.AssertionClassifierGrpc.*;
import edu.uw.bhi.bionlp.covid.parser.CovidParser.AssertionClassifierInput;
import edu.uw.bhi.bionlp.covid.parser.CovidParser.AssertionClassifierOutput;
import io.grpc.Channel;
import io.grpc.ManagedChannelBuilder;

public class AssertionClassifierChannelManager {

    private final Channel channel;

    public AssertionClassifierChannelManager() {

        this.channel = ManagedChannelBuilder
            .forAddress("assertion-classifier", Integer.parseInt(System.getenv("ASRTCLA_PORT")))
            .usePlaintext(true)
            .build();
    }

    public AssertionClassifierClient generateClient() {

        return new AssertionClassifierClient(this.channel);
    }

    public class AssertionClassifierClient {

        private final AssertionClassifierBlockingStub stub;

        public AssertionClassifierClient(Channel channel) {
            this.stub = AssertionClassifierGrpc.newBlockingStub(channel);
        }

        public String predictAssertion(String sentence, int beginIndex, int endIndex) {

            AssertionClassifierOutput output;
            AssertionClassifierInput request = AssertionClassifierInput.newBuilder()
                .setText(sentence)
                .setStartCharIndex(beginIndex)
                .setEndCharIndex(endIndex)
                .build();
    
            try {
                output = this.stub.predictAssertion(request);
            } catch (Exception ex) {
                return "present";
            }
            return output.getPrediction();
        }
    }
}