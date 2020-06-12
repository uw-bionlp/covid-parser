package edu.uw.bhi.bionlp.utils;

import edu.uw.bhi.uwassert.AssertionClassification;
import java.io.IOException;
import name.adibejan.util.IntPair;

/**
 *
 * @author wlau
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
    public String predict (String sentence, int startIndex, int endIndex) throws IOException {
	    return AssertionClassification.predictMP(sentence, new IntPair(startIndex, endIndex));  
    }

    public static String Present = "present";
}