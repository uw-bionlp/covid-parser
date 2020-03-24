package edu.uw.bhi.bionlp;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;

/**
 *
 * @author wlau
 * @author ndobb
 */
public class SentenceChunker {

    SentenceDetectorME sentenceDetector;

    public SentenceChunker() {

        try {
            InputStream modelIn = new FileInputStream("resources/opennlp/en-sent.bin");
            SentenceModel model = new SentenceModel(modelIn);
            sentenceDetector = new SentenceDetectorME(model);
        } catch (Exception ex) {
            System.out.println(ex.getStackTrace());
            System.out.println(ex.getMessage());
        }
    }

    public List<String> getSentences(String text) throws Exception{
        
        return Arrays.asList(sentenceDetector.sentDetect(text));
    }
}