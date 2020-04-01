package edu.uw.bhi.bionlp.utils;

public class Token {
    String text;
    int index;

    public Token() {}
    public Token(String text, int index) {
        this.text = text;
        this.index = index;
    }

    public String getText() {
        return this.text;
    }
    public void setText(String text) {
        this.text = text;
    }
    public int getIndex() {
        return this.index;
    }
    public void setIndex(int index) {
        this.index = index;
    }
    public boolean contains(String str) {
        return this.text.contains(str);
    }
    public int length() {
        return this.text.length();
    }
}