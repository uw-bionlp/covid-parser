package edu.uw.bhi.bionlp.covid.parser;

import java.util.ArrayList;
import java.util.List;

import edu.uw.bhi.bionlp.covid.parser.CovidParser.Sentence;
import edu.uw.bhi.bionlp.covid.parser.CovidParser.SentenceDetectionInput;
import edu.uw.bhi.bionlp.covid.parser.CovidParser.SentenceDetectionOutput;
import edu.uw.bhi.bionlp.covid.parser.CovidParser.Sentence.Builder;
import edu.uw.bhi.bionlp.covid.parser.OpenNLPGrpc.OpenNLPImplBase;
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
            int i = 0;
            List<Span> spans = sentenceDetector.detectBoundaries(text);
            for (Span span : spans) {
                int start = span.getStart();
                int end = span.getEnd();
                String sentText = end == lastCharIdx ? text.substring(start) : text.substring(start, end);
                Sentence sentence = sentBuilder
                    .setBeginCharIndex(start)
                    .setEndCharIndex(end)
                    .setText(sentText)
                    .setId(i)
                    .build();
                sentences.add(sentence);
                i++;
            }
        }

        SentenceDetectionOutput response = SentenceDetectionOutput
            .newBuilder()
            .addAllSentences(sentences)
            .setText(text)
            .setId(input.getId())
            .build();
        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }
}