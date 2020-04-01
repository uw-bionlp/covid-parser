package edu.uw.bhi.bionlp.covid.parser;

import java.util.ArrayList;
import java.util.List;

import edu.uw.bhi.bionlp.covid.parser.assertionclassifier.AssertionClassifierGrpc.AssertionClassifierImplBase;
import edu.uw.bhi.bionlp.covid.parser.assertionclassifier.AssertionClassifierOuterClass.*;
import edu.uw.bhi.bionlp.covid.parser.assertionclassifier.AssertionClassifierOuterClass.Sentence.Builder;
import edu.uw.bhi.uwassert.AssertionClassification;
import io.grpc.stub.StreamObserver;

import name.adibejan.util.IntPair;
import opennlp.tools.util.Span;

/**
 *
 * @author ndobb
 */
public class AssertionClassifierImpl extends AssertionClassifierImplBase {

    SentenceDetector sentenceDetector = new SentenceDetector();

    static {
        System.setProperty("CONFIGFILE",
                    "resources/assertion-classifier/assertcls.properties");
        System.setProperty("ASSERTRESOURCES",
                    "resources/assertion-classifier");
        System.setProperty("LIBLINEAR_PATH",
                    "resources/assertion-classifier/liblinear-1.93");
    }

    @Override
    public void predictAssertion(AssertionClassifierInput input, StreamObserver<AssertionClassifierOutput> responseObserver) {

        String prediction = predict(input.getText(), input.getStartIndex(), input.getEndIndex());
        AssertionClassifierOutput response = AssertionClassifierOutput
            .newBuilder()
            .setPrediction(prediction)
            .build();
        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }

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

    String predict (String sentence, int startIndex, int endIndex) {
        try {
            return AssertionClassification.predictMP(sentence, new IntPair(startIndex, endIndex));  
        } catch (Exception ex) {
            return "present";
        }
    }
}