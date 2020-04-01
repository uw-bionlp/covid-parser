package edu.uw.bhi.bionlp.predictor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import edu.uw.bhi.bionlp.utils.*;

public class CovidResult {
    public List<CovidSentence> sentences = new ArrayList<CovidSentence>();
    public String text;
    public String prediction;

    public CovidResult(String text) {
        this.text = text;
    }

	public boolean hasMatches() { 
        for (CovidSentence sentence : sentences) {
            if (sentence.hasMatches()) {
                return true;
            }
        }
        return false;
    }

    public void resolvePrediction() {
        HashMap<String,Double> scores = new HashMap<String,Double>();
        for (CovidSentence sentence : sentences) {
            for (CovidConcept con : sentence.concepts) {
                if (scores.containsKey(con.prediction)) {
                    scores.put(con.prediction, scores.get(con.prediction) + con.covidPhrase.getWeight());
                } else {
                    scores.put(con.prediction, con.covidPhrase.getWeight());
                }
            }
        }
        Double max = 0.0;
        for (String k : scores.keySet()) {
            if (scores.get(k) > max) {
                max = scores.get(k);
                this.prediction = k;
            }
        }
    }
}

class CovidSentence {
    public int index;
    public String text;
    public List<Token> tokens = new ArrayList<Token>();
    public List<CovidConcept> concepts = new ArrayList<CovidConcept>();
    public String prediction;

    public CovidSentence(int index, String text) {
        this.index = index;
        this.text = text;
        this.tokens = Tokenizer.tokenize(text);
    }

    public boolean hasMatches() { 
        return concepts.size() > 0; 
    }

    public boolean hasDisagreement() {
        if (concepts.size() > 0) {
            String firstPrediction = concepts.get(0).prediction;
            for (CovidConcept con : concepts) {
                if (con.prediction != firstPrediction) {
                    return true;
                }
            }
        }
        return false;
    }

    public void resolvePrediction() {
        HashMap<String,Double> scores = new HashMap<String,Double>();

        for (CovidConcept con : concepts) {
            if (scores.containsKey(con.prediction)) {
                scores.put(con.prediction, scores.get(con.prediction) + con.covidPhrase.getWeight());
            } else {
                scores.put(con.prediction, con.covidPhrase.getWeight());
            }
        }
        Double max = 0.0;
        for (String k : scores.keySet()) {
            if (scores.get(k) > max) {
                max = scores.get(k);
                this.prediction = k;
            }
        }
    }
}

class CovidConcept {
    public WeightedSearchTerm covidPhrase;
    public Token token;
    public List<Token> sentenceTokens;
    public String prediction;
    public AssertionClassifierNgramParams params;

    public CovidConcept(WeightedSearchTerm covidPhrase, Token token, List<Token> sentenceTokens) {
        this.covidPhrase = covidPhrase;
        this.token = token;
        this.sentenceTokens = sentenceTokens;
        this.params = getNgram();
    }

    AssertionClassifierNgramParams getNgram() {
        int size = 5;
        int main = this.token.getIndex();
        int idx = main-size > 0 ? main-size : 0;
        int beginIdx = 0;
        int endIdx = 0;
        String sep = " ";
        StringBuilder sb = new StringBuilder();

        // Preceding
        while (idx < main) {
            sb.append(this.sentenceTokens.get(idx).getText() + sep);
            idx++;
            beginIdx++;
        }

        // Main token
        sb.append(this.sentenceTokens.get(main).getText() + sep);
        endIdx = beginIdx;
        idx = main+1;

        // Following
        while (idx <= main+size && idx < this.sentenceTokens.size()) {
            sb.append(this.sentenceTokens.get(idx).getText() + sep);
            idx++;
        }
        return new AssertionClassifierNgramParams(sb.toString(), beginIdx, endIdx);
    }

    class AssertionClassifierNgramParams {
        public String ngram;
        public int beginIndex;
        public int endIndex;

        public AssertionClassifierNgramParams(String ngram, int beginIndex, int endIndex) {
            this.ngram = ngram;
            this.beginIndex = beginIndex;
            this.endIndex = endIndex;
        }
    }
}

    