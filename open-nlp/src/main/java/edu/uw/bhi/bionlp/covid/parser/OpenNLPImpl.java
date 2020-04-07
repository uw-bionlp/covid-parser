package edu.uw.bhi.bionlp.covid.parser;

import java.util.ArrayList;
import java.util.List;

import edu.uw.bhi.bionlp.covid.parser.opennlp.OpenNLPGrpc.OpenNLPImplBase;
import edu.uw.bhi.bionlp.covid.parser.opennlp.OpenNLPOuterClass.Sentence;
import edu.uw.bhi.bionlp.covid.parser.opennlp.OpenNLPOuterClass.SentenceDetectionInput;
import edu.uw.bhi.bionlp.covid.parser.opennlp.OpenNLPOuterClass.SentenceDetectionOutput;
import edu.uw.bhi.bionlp.covid.parser.opennlp.OpenNLPOuterClass.Sentence.Builder;
import io.grpc.stub.StreamObserver;
import opennlp.tools.util.Span;

/**
 *
 * @author ndobb
 */
public class OpenNLPImpl extends OpenNLPImplBase {

    SentenceDetector sentenceDetector = new SentenceDetector();

    @Override
    public void detectSentences(SentenceDetectionInput input, StreamObserver<SentenceDetectionOutput> responseObserver) {

        List<Sentence> sentences = new ArrayList<Sentence>();
        Builder sentBuilder = Sentence.newBuilder();
        String text = input.getText();
        int textLen = text.length();

        if (textLen > 0) {
            int lastCharIdx = textLen-1;
            List<Span> spans = sentenceDetector.detectBoundaries(input.getText());
            for (Span span : spans) {
                int start = span.getStart();
                int end = span.getEnd();
                String sentText = end == lastCharIdx ? text.substring(start) : text.substring(start, end);
                Sentence sentence = sentBuilder
                    .setBeginCharIndex(start)
                    .setEndCharIndex(end)
                    .setText(sentText)
                    .build();
                sentences.add(sentence);
            }
        }

        SentenceDetectionOutput response = SentenceDetectionOutput
            .newBuilder()
            .addAllSentences(sentences)
            .build();
        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }
}