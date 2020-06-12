package edu.uw.bhi.bionlp.predictor;

public class WeightedSearchTerm {
    String text;
    Double weight;

    public WeightedSearchTerm(String term, Double weight) {
        this.text = term;
        this.weight = weight;
    }

    public String getText() {
        return this.text;
    }
    public void setText(String text) {
        this.text = text;
    }
    public Double getWeight() {
        return this.weight;
    }
    public void setWeight(Double weight) {
        this.weight = weight;
    }
}