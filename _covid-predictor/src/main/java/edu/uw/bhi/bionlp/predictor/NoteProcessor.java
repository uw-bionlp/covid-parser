package edu.uw.bhi.bionlp.predictor;

import java.util.List;

import edu.uw.bhi.bionlp.utils.*;
import opennlp.tools.util.Span;

import java.util.ArrayList;
import java.util.Arrays;

public class NoteProcessor {

    SentenceChunker sentenceChunker = new SentenceChunker();
    AssertionClassifier classifier = new AssertionClassifier();
    Tokenizer tokenizer = new Tokenizer();

    ArrayList<WeightedSearchTerm> terms = new ArrayList<WeightedSearchTerm>(
        Arrays.asList(
            new WeightedSearchTerm("coronavirus", 0.5), 
            new WeightedSearchTerm("sars-cov", 1.0), 
            new WeightedSearchTerm("sars-2", 1.0),
            new WeightedSearchTerm("covid", 1.0)
    ));

    public CovidResult processNote(String note) throws Exception {

        CovidResult result = new CovidResult(note);
        List<String> sentences = sentenceChunker.getSentences(note);
        List<Span> x = sentenceChunker.detectBoundaries(note);

        for (int sId = 0; sId < sentences.size(); sId++) {
            CovidSentence sentence = new CovidSentence(sId, sentences.get(sId));

            for (WeightedSearchTerm term : terms) {
                for (Token tok : sentence.tokens) {
                    if (tok.contains(term.text)) {
                        CovidConcept con = new CovidConcept(term, tok, sentence.tokens);

                        // Default assumption is condition is present
                        con.prediction = AssertionClassifier.Present;
                        try {
                            con.prediction = classifier.predict(con.params.ngram, con.params.beginIndex, con.params.endIndex);
                        } catch (Exception ex) {
                            System.out.println("Error in assertion classifier: " + ex.toString());
                        }
                        sentence.concepts.add(con);
                    }
                }
            }
            if (sentence.hasMatches()) {
                sentence.resolvePrediction();
                result.sentences.add(sentence);
            }
        }
        return result;
    }
}

