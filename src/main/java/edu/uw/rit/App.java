package edu.uw.rit;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class App 
{
    public static void main( String[] args )
    {
        NoteProcessor processor = new NoteProcessor();
        List<CovidResult> results = new ArrayList<CovidResult>();

        for (final File file : new File("resources/examples").listFiles()) {
            try {
                if (!file.getPath().endsWith("txt")) {
                    continue;
                }
                String note = new String(Files.readAllBytes(file.toPath()));
                results.add(processor.processNote(note));

            } catch (Exception ex) { 
                // Do nothing
            }
        }

        System.out.println("");
        System.out.println("************************");
        System.out.println("* Results");
        System.out.println("************************");

        int n = 1;
        for (CovidResult res : results) {
            System.out.println("");
            System.out.println("Note " + n);
            System.out.println("Text: '" + res.text + "'");
            int s = 1;
            for (CovidSentence sent : res.sentences) {
                System.out.println("    Sentence " + s);
                System.out.println("    Prediction: '" + sent.prediction + "'");
                s++;
                int c = 1;
                for (CovidConcept con : sent.concepts) {
                    System.out.println("        Concept " + c);
                    System.out.println("        Context:    '" + con.params.ngram + "'");
                    System.out.println("        Phrase:     '" + con.covidPhrase.getText() + "'");
                    System.out.println("        Prediction: '" + con.prediction + "'");
                    c++;
                }
            }
            n++;
        }
    }
}
