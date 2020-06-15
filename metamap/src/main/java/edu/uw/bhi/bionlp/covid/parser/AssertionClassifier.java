package edu.uw.bhi.bionlp.covid.parser;

import java.io.File;

import edu.uw.bhi.uwassert.AssertionClassification;
import name.adibejan.util.IntPair;
import org.javatuples.Pair;

/**
 *
 * @author ndobb
 */
public class AssertionClassifier {
    static {
        System.setProperty("CONFIGFILE",
                    "resources/assertion-classifier/assertcls.properties");
        System.setProperty("ASSERTRESOURCES",
                    "resources/assertion-classifier");
        System.setProperty("LIBLINEAR_PATH",
                    "resources/assertion-classifier/liblinear-1.93");
    }

    public AssertionClassifier() {
        clearFeatureCache();
    }

    Pair<String,String> predict (String sentence, int startIndex, int endIndex) {
        String prediction = "indeterminate";
        String ngram = "";
        
        try {
            NgramParameters params = getNgram(sentence, 5, startIndex, endIndex);
            ngram = params.ngram;
            prediction = AssertionClassification.predictMP(ngram, new IntPair(params.beginTokenIndex, params.endTokenIndex));  
            return new Pair<String,String>(prediction, null);
        } catch (Exception ex) {
            String err = "Error: failed to assert. NGram: " + ngram + ", StartIndex: " + startIndex + ", EndIndex: " + endIndex + ". Error: " + ex.getMessage();
            clearFeatureCache();
            return new Pair<String,String>(prediction, err);
        }
    }

    void clearFeatureCache() {
        File[] files = new File("resources/assertion-classifier/features").listFiles();
        for (int i = 0; i < files.length; i++) {
            File file = files[i];
            String name = file.getName();
            if (name.startsWith("runtext.feat_") || name.startsWith("runtext.predict_")) {
                try {
                    file.delete();
                } catch (Exception ex) {
                    // do nothing  
                }
            }
        }
    }

    NgramParameters getNgram(String sentence, int ngramSize, int beginCharIndex, int endCharIndex) {
        int lastChar = sentence.length()-1;
        int precCnt = 0;
        int follCnt = 0;
        int startPos = beginCharIndex;
        int endPos = endCharIndex;
        boolean isSpace = false;
        boolean prevWasSpace = true;
        String sent = sentence.replaceAll("\n", " ");

        // Get preceding
        while (startPos > -1) {
            isSpace = sent.charAt(startPos) == ' ';
            if (isSpace && !prevWasSpace) {
                precCnt++;
            }
            if (startPos == 0 || precCnt > ngramSize) {
                if (precCnt > ngramSize) {
                    precCnt = ngramSize;
                }
                break;
            }
            startPos--;
            prevWasSpace = isSpace;
        }

        // Get following
        prevWasSpace = true;
        while (endPos <= lastChar) {
            isSpace = sent.charAt(endPos) == ' ';
            if (isSpace && !prevWasSpace) {
                follCnt++;
            }
            if (endPos == lastChar || follCnt == ngramSize) {
                break;
            }
            endPos++;
            prevWasSpace = isSpace;
        }

        return new NgramParameters(
            sent.substring(startPos, endPos).trim(),
            precCnt, 
            precCnt + sent.substring(beginCharIndex, endCharIndex).split(" ").length-1
        );
    }
}

class NgramParameters {
    public String ngram;
    public int beginTokenIndex;
    public int endTokenIndex;

    public NgramParameters(String ngram, int beginTokenIndex, int endTokenIndex) {
        this.ngram = ngram;
        this.beginTokenIndex = beginTokenIndex;
        this.endTokenIndex = endTokenIndex;
    }
}              