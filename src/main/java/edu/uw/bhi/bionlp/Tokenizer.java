package edu.uw.bhi.bionlp;

import java.util.ArrayList;
import java.util.List;

/**
 * Author: melihay, ndobb
 * Date: 7/10/15, adapted 3/24/20
 * Version: 1.0
 */
public class Tokenizer {

    public static List<Token> tokenize(String text) {
        List<Token> tokenList = new ArrayList<Token>();

        String result = text.replaceAll("[!\"#$%&'()*,./:;<=>?\\\\@\\[\\]^_`{|}~]", " ");
        String trimresult = result.replaceAll("\\s+", " ").trim();

        if (!trimresult.equals("")) {
            String[] tokens = trimresult.split("\\s+");
            for (int i = 0; i < tokens.length; i++) {
                String contentWord = tokens[i].toLowerCase();
                Token tok = new Token(contentWord, i);
                tokenList.add(tok);
            }
        }
        return tokenList;
    }
}