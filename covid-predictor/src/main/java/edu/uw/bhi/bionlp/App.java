package edu.uw.bhi.bionlp;

import edu.uw.bhi.bionlp.predictor.CovidWorkerImpl;
import io.grpc.*;

public class App 
{
    public static void main( String[] args ) throws Exception
    {
      // Create a new server to listen on port 8080
      Server server = ServerBuilder.forPort(8080)
        .addService(new CovidWorkerImpl())
        .build();

      // Start the server
      server.start();

      // Server threads are running in the background.
      System.out.println("Server is up!");

      // Don't exit the main thread. Wait until server is terminated.
      server.awaitTermination();
    }
}

/*
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
                CovidResult result = processor.processNote(note);
                result.resolvePrediction();
                results.add(result);

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
*/